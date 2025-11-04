import argparse
import hashlib
import math
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

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
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Eagle3 model")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    parser.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )

    # add evaluation-related arguments
    parser.add_argument("--train-data-path", type=str, required=True) # for vocab mapping
    parser.add_argument("--eval-data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--ttt-length", type=int, default=7)

    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )

    # distributed evaluation
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument("--attention-backend", type=str, default="flex_attention")

    # vlm related args
    parser.add_argument(
        "--min-pixels", type=int, default=50176
    )  # 64 * 28 * 28 for qwen2.5-vl
    parser.add_argument(
        "--max-pixels", type=int, default=802816
    )  # 1024 * 28 * 28 for qwen2.5-vl

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    return parser, args


def main():
    # initialize
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed environment")
    args.dp_size = dist.get_world_size() // args.tp_size

    # Handle draft model config
    if args.draft_model_config is None:
        # Auto-generate and save config file
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=args.cache_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        # Use provided config file
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # build target model
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
        if args.is_vlm and draft_model_config.target_model_type == "qwen2_5_vl":
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

    for p in target_model.parameters():
        p.requires_grad = False

    print_with_rank("Initialized target model")

    # load trained draft model
    draft_model = AutoEagle3DraftModel.from_pretrained(
        args.checkpoint_path,
        attention_backend=args.attention_backend,
        torch_dtype=torch.bfloat16,
    ).cuda()
    
    # 加载embedding和vocab mapping
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank("Loaded trained draft model")

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    else:
        processor = None

    # convert to dataloader
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    vocab_cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    
    cache_params_string = (
        f"{args.eval_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
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
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=eval_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=vocab_cache_key,
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

    # load vocab mapping
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    # build Eagle3 model
    if args.is_vlm and draft_model_config.target_model_type == "qwen2_5_vl":
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
        eagle3_model = OnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=[target_model],
        process_group=get_dp_group(),
    )
    print_with_rank("Initialized Eagle3 FSDP model")

    # run evaluation
    draft_model.eval()
    eval_acces = [[] for _ in range(eagle3_model.module.length)]
    eval_plosses = [[] for _ in range(eagle3_model.module.length)]

    if dist.get_rank() == 0:
        progress_bar = tqdm(eval_dataloader, desc="Evaluating", leave=True)
    else:
        progress_bar = eval_dataloader

    last_time = time.time()
    total_samples = 0

    for data in progress_bar:
        if args.is_vlm:
            with torch.no_grad():
                plosses, _, acces = eagle3_model.evaluation(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                    # sparse_attention_mask=data["sparse_attention_mask"].cuda(),
                    pixel_values=data["pixel_values"].cuda(),
                    image_grid_thw=data["image_grid_thw"].cuda(),
                )
        else:
            with torch.no_grad():
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                )
        
        acces = torch.stack(acces).cpu().tolist()

        eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
        eval_plosses = [
            eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
        ]

        total_samples += data["input_ids"].shape[0]

        if args.verbose:
            print(
                f"[{dist.get_rank()}] time={(time.time() - last_time):.3}s shape={data['input_ids'].shape}"
            )
            last_time = time.time()

        if dist.get_rank() == 0:
            avg_loss = sum(pl.item() for pl in plosses) / len(plosses)
            avg_acc = sum(acces) / len(acces)
            progress_bar.set_postfix(
                {"loss": f"{avg_loss:.2f}", "acc": f"{avg_acc:.2f}", "samples": total_samples}
            )

    # Synchronize and collect results from all devices
    all_acces = []
    all_plosses = []
    
    for i in range(len(eval_acces)):
        # Synchronize accuracy
        acc_i = torch.tensor(eval_acces[i]).cuda()
        acc_mean = acc_i.mean()
        dist.all_reduce(acc_mean)
        acc_mean = (acc_mean / dist.get_world_size()).item()
        
        # Synchronize loss
        loss_i = torch.tensor(eval_plosses[i]).cuda()
        loss_mean = loss_i.mean()
        dist.all_reduce(loss_mean)
        loss_mean = (loss_mean / dist.get_world_size()).item()
        
        all_acces.append(acc_mean)
        all_plosses.append(loss_mean)

    # Synchronize total samples
    total_samples_tensor = torch.tensor(total_samples).cuda()
    dist.all_reduce(total_samples_tensor)
    total_samples = total_samples_tensor.item()

    # Only rank 0 prints the results
    if dist.get_rank() == 0:
        # Calculate overall metrics
        overall_acc = sum(all_acces) / len(all_acces)
        overall_loss = sum(all_plosses) / len(all_plosses)

        # Print results in table format
        print("\n" + "="*70)
        print("EVALUATION RESULTS (Synchronized across all devices)")
        print("="*70)
        print(f"{'Position':<10} {'Accuracy':<12} {'Loss':<12}")
        print("-"*70)
        
        for i in range(len(all_acces)):
            print(f"{i:<10} {all_acces[i]:<12.4f} {all_plosses[i]:<12.4f}")
        
        print("-"*70)
        print(f"{'OVERALL':<10} {overall_acc:<12.4f} {overall_loss:<12.4f}")
        print("="*70)
        print(f"Total samples evaluated: {total_samples}")
        print("="*70)

    destroy_distributed()


if __name__ == "__main__":
    main()