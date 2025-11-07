#!/bin/bash
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
echo $ROOT_DIR

# support tp1 train eagle3 for qwen2.5-vl-7b-instruct
NUM_GPUS=1
echo "using gpu num: $NUM_GPUS"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/evaluation/run_base.py \
    --target-model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2-5-vl-7b-eagle3.json \
    --checkpoint-path /home/wangrunqi03/eagle_vlm_dataset/saved_models/qwen_v46_save_model/epoch_3/ \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v_qwen2_5_vl_clean_train.jsonl \
    --eval-data-path $ROOT_DIR/cache/dataset/allava4v_qwen2_5_vl_test.jsonl \
    --max-length 8192 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --attention-backend flex_attention \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --batch-size 1 \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --verbose
