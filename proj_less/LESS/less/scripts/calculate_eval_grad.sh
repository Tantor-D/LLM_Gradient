# 运行这一段代码前需要注意看看需不需要进入代码中把sample的方式改了


# 定义CKPT数组
# CKPTS=(13 27 41 52)
CKPTS=(4 8 12 16)


# 循环遍历每个CKPT
for CKPT in "${CKPTS[@]}"
do
    # 定义并处理每个训练数据集
    for TASK in "AQuA" "GSM"
    do
        SEED=8
        MODEL_PATH=../out/llama2-7b-p0.05-lora-seed${SEED}/checkpoint-${CKPT}
        OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed${SEED}/${TASK}-ckpt${CKPT}-sgd
        DATA_DIR=../data
        DIMS="8192"

        # 执行数据处理脚本
        echo "Processing $TASK at checkpoint $CKPT..."
        ./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"

        # 打印状态消息
        echo "Processed testing data for CKPT=$CKPT with TASK=$TASK"
    done
done


