# draft model decoding 长度
# target model 验证 （tree mask）
# 计算accept length
# target model的输出已经有了，主要就是一个验证的pipeline问题
# 优先完成QwenVL 2.5 的验证

# QwenVL2.5路径
import argparse
import hashlib
import math
import os
import time
import torch
from collections import defaultdict
from accelerate.utils import set_seed
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType

from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
)

from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_device_mesh,
    init_distributed,
)
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)

from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument(
        "--draft-model-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument("--attention-backend", type=str, default="flex_attention")

    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--seed", type=int, default=0)

    # vlm related args
    parser.add_argument(
        "--min-pixels", type=int, default=50176
    )  # 64*28*28 for qwen2.5-vl
    parser.add_argument(
        "--max-pixels", type=int, default=802816
    )  # 1024*28*28 for qwen2.5-vl
    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )
    args = parser.parse_args()

    return parser, args

parser, args = parse_args()

init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
print_with_rank("Initialized distributed environment")


set_seed(args.seed)

config = AutoConfig.from_pretrained(args.draft_model_path)
draft_model = LlamaForCausalLMEagle3.from_pretrained(
    args.draft_model_path,
    torch_dtype=config.torch_dtype
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
if args.is_vlm:
    processor = AutoProcessor.from_pretrained(
        args.target_model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
else:
    processor = None

if args.tp_size > 1:
    # check if the target model has tp_plan
    config = AutoConfig.from_pretrained(args.target_model_path)

    if type(config) in AutoDistributedTargetModel._model_mapping:
        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=args.target_model_path,
            torch_dtype=torch.bfloat16,
            device="cuda",
            local_files_only=True,
        ).eval()
    else:
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            tp_plan="auto",
            tp_size=args.tp_size,
            torch_dtype=torch.bfloat16,
            device_mesh=get_tp_device_mesh(),
        ).eval()
else:
    if args.is_vlm and draft_model.config.target_model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        target_model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .cuda()
        )
    else:
        target_model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
                cache_dir=args.cache_dir,
            )
            .eval()
            .cuda()
        )

if args.is_vlm and draft_model.config.target_model_type == "qwen2_5_vl":
    eagle3_model = QwenVLOnlineEagle3Model(
        target_model=target_model,
        draft_model=draft_model,
        processor=processor,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
    ).eval().cuda()
else:
    eagle3_model = OnlineEagle3Model(
        target_model=target_model,
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
    ).eval().cuda()

cache_params_string = (
        f"{args.eval_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )

cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]

with rank_0_priority():
    eval_eagle3_dataset = build_eagle3_dataset(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        is_vlm=args.is_vlm,
        is_preformatted=args.is_preformatted,
        processor=processor,
        num_proc=args.build_dataset_num_proc,
    )

eval_dataloader = prepare_dp_dataloaders(
    eval_eagle3_dataset,
    args.batch_size,
    num_workers=4,
    shuffle=False,
    process_group=get_dp_group(),
    is_vlm=args.is_vlm,
)
print_with_rank("Initialized eval dataloader")

# 2. target model prefill得到 hidden_states

for data in tqdm(eval_dataloader, desc=f"Evaluating Accepth Length"):
    if args.is_vlm:
        with torch.no_grad():
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
                pixel_values=data["pixel_values"].cuda(),
                image_grid_thw=data["image_grid_thw"].cuda(),
            )
        print(acces)

exit()


# 3. draft model 的自回归解码 length长度的token

# 4. 和验证集truth token进行check，计算接受率，并且得到新的 hidden_states

# 5. 进行下一次draft model的自回归解码
