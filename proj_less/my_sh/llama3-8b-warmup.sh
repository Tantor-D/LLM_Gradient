#!/bin/bash
#^llama3-8b-warmup
#&要去实际运行的shell处修改torchrun调用几个显卡

# 环境配置
pwd
chmod +x ./less/scripts/train/warmup_lora_train.sh
# export CUDA_VISIBLE_DEVICES=1

# warm up training
DATA_DIR=../data
MODEL_PATH=/root/xinglin-data/dsw_data/pretrain_model/Llama-3-8b
PERCENTAGE=0.05
DATA_SEED=3
JOB_NAME=llama3-8b-p${PERCENTAGE}-lora-seed${DATA_SEED}
./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"

