#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp1 evaluate eagle3 for qwen2.5-vl-7b-instruct
NUM_GPUS=1
# allava4v_qwen2_5_vl_test.jsonl
# test_kling_vl.jsonl
CHECKPOINT_PATH=$ROOT_DIR/outputs/Qwen2.5-VL-7B-eagle3-dsa/epoch_7
DRAFT_MODEL_CONFIG=$ROOT_DIR/configs/qwen2-5-vl-7b-eagle3-dsa.json
ATTENTION_BACKEND=dsa
# ATTENTION_BACKEND=sdpa

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/eval_eagle3.py \
    --target-model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --checkpoint-path $CHECKPOINT_PATH \
    --eval-data-path $ROOT_DIR/cache/dataset/test_kling_vl.jsonl \
    --max-length 8192 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --attention-backend $ATTENTION_BACKEND \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --batch-size 1 \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --verbose
