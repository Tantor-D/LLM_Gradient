# 环境配置这一块
pwd
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

        if [[ ! -d $OUTPUT_PATH ]]; then
          mkdir -p $OUTPUT_PATH
        fi

        # 定义日志文件
        LOG_FILE="${OUTPUT_PATH}/processing.log"

        # 执行数据处理脚本
        # ! 注意这里是我自己写的代码
        echo "Processing $TRAINING_DATA_NAME at checkpoint $CKPT..."
        python3 -m less.data_selection.my_get_info \
        --train_file $TRAINING_DATA_FILE \
        --info_type grads \
        --model_path $MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --gradient_projection_dimension $DIMS \
        --gradient_type $GRADIENT_TYPE  2>&1 | tee -a $LOG_FILE

        # 打印状态消息
        echo "Processed training data for CKPT=$CKPT with DATA NAME=$TRAINING_DATA_NAME"

    done
done