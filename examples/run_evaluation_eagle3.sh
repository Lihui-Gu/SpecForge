#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp1 train eagle3 for qwen2.5-vl-7b-instruct
NUM_GPUS=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/evaluation/run_eagle3_evaluation.py \
    --target-model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --draft-model-path $ROOT_DIR/outputs/Qwen2.5-VL-7B-eagle3/epoch_0 \
    --eval-data-path $ROOT_DIR/cache/dataset/test_kling_vl.jsonl \
    --max-length 16384 \
    --chat-template qwen2-vl \
    --cache-dir $ROOT_DIR/cache \
    --tp-size 1 \
    --is-vlm \
    --attention-backend sdpa \
    --min-pixels 50176 \
    --max-pixels 802816
