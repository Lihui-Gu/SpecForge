import argparse
import ast
import os
import re
import shutil
import time
import json
from datasets import load_dataset
from sglang import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)

def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))
    """
    cache_dir = os.path.join(".cache", "mmstar_specforge")
    image_dir = os.path.join(cache_dir, "images")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    print(f"Created temporary image directory: {cache_dir}")
    """

    # Read data
    data_files = "/home/gulihui/workspace/dev_github/SpecForge/cache/dataset/test_kling_vl.jsonl"
    dataset = load_dataset("json", data_files=data_files)["train"]

    questions = []
    for idx, q in enumerate(dataset):
        if idx >= args.num_questions:
            break
        item = [
            {
                "image_path": q["image"],
                "system":  q["conversations"][0]["content"].strip(),
                "question": q["conversations"][1]["content"].strip(),
            }
        ]
        questions.append(item)
    #####################################
    ######### SGL Program Begin #########
    #####################################
    import sglang as sgl

    @sgl.function
    def get_kling_answer(s, question):
        s += sgl.system(question["system"])
        s += sgl.user(sgl.image(question["image_path"]) + question["question"])
        s += sgl.assistant(sgl.gen("answer"))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = get_kling_answer.run_batch(
        questions,
        temperature=0,
        max_new_tokens=2048,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )

    output_throughput = num_output_tokens / latency

    has_verify = "spec_verify_ct" in states[0].get_meta_info("answer")
    if has_verify:
        num_verify_tokens = sum(
            s.get_meta_info("answer")["spec_verify_ct"] for s in states
        )
        if num_verify_tokens == 0:
            accept_length = 1.0
        else:
            accept_length = num_output_tokens / num_verify_tokens
    else:
        accept_length = 1.0

    # Print results
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Accept length: {accept_length:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=20)
    args = add_common_sglang_args_and_parse(parser)
    main(args)