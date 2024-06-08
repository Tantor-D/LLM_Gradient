#!/bin/bash
#^llama2-7b-warmup


# 环境配置
pwd
chmod +x ./less/scripts/train/warmup_lora_train.sh
export CUDA_VISIBLE_DEVICES=0


# warm up training
DATA_DIR=../data
MODEL_PATH=/root/xinglin-data/dsw_data/pretrain_model/Llama-2-7b-hf
PERCENTAGE=0.05
DATA_SEED=3
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}
./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"

