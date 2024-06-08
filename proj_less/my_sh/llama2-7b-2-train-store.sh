#!/bin/bash
#^llama2-7b-train-store

# 环境配置这一块
pwd
chmod +x ./less/scripts/get_info/grad/get_train_lora_grads.sh 
export CUDA_VISIBLE_DEVICES=0


CKPTS=(4 8 12 16)
seed=3
GRADIENT_TYPE="adam"
DIMS="8192"


# 循环遍历每个CKPT
for CKPT in "${CKPTS[@]}"
do
    # 定义并处理每个训练数据集
    for TRAINING_DATA_NAME in "code_high" "code_medium" "code_low"
    do
        # 根据训练数据名称设置对应的文件路径
        if [ "$TRAINING_DATA_NAME" == "code_high" ]; then
            TRAINING_DATA_FILE="../data/train/processed/code_high/code_high_data.jsonl"
        elif [ "$TRAINING_DATA_NAME" == "code_medium" ]; then
            TRAINING_DATA_FILE="../data/train/processed/code_medium/code_medium_data.jsonl"
        elif [ "$TRAINING_DATA_NAME" == "code_low" ]; then
            TRAINING_DATA_FILE="../data/train/processed/code_low/code_low_data.jsonl"
        fi

        # 定义其他变量
        MODEL_PATH="../out/llama2-7b-p0.05-lora-seed${seed}/checkpoint-${CKPT}"
        OUTPUT_PATH="../grads/llama2-7b-p0.05-lora-seed${seed}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}"
        

        # 执行数据处理脚本
        echo "Processing $TRAINING_DATA_NAME at checkpoint $CKPT..."
        ./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"

        # 打印状态消息
        echo "Processed training data for CKPT=$CKPT with DATA NAME=$TRAINING_DATA_NAME"
    done
done