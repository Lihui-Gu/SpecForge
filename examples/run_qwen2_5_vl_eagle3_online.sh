#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export CUDA_VISIBLE_DEVICES=4,5,6,7
# support tp1 train eagle3 for qwen2.5-vl-7b-instruct
NUM_GPUS=4

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2-5-vl-7b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v_qwen2_5_vl_clean_train.jsonl \
    --eval-data-path $ROOT_DIR/cache/dataset/allava4v_qwen2_5_vl_test.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen2.5-VL-7B-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --draft-global-batch-size 8 \
    --draft-micro-batch-size 2 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --attention-backend sdpa \
    --tp-size 1 \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --report-to wandb
