#!/bin/bash
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
echo $ROOT_DIR

# support tp1 train eagle3 for qwen2.5-vl-7b-instruct
NUM_GPUS=8
echo "using gpu num: $NUM_GPUS"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/evaluation/run_base.py \
    --target-model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2-5-vl-7b-eagle3.json \
    --checkpoint-path /home/gulihui/workspace/dev_github/SpecForge/outputs/Qwen2.5-VL-7B-eagle3/epoch_2 \
    --train-data-path $ROOT_DIR/cache/dataset/train_kling_vl.jsonl \
    --eval-data-path $ROOT_DIR/cache/dataset/test_kling_vl.jsonl \
    --max-length 8192 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --attention-backend sdpa \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --verbose
