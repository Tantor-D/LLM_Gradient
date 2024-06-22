# !运行这一段代码前需要注意看看需不需要进入代码中把sample的方式改了

pwd
chmod +x ./less/scripts/get_info/grad/get_eval_lora_grads.sh
export CUDA_VISIBLE_DEVICES=1

# 定义CKPT数组
CKPTS=(4 8 12 16)
MODEL_NAME="llama2-7b"
SEED=3
DIMS="8192"
DATA_DIR=../data


# 循环遍历每个CKPT
for CKPT in "${CKPTS[@]}"
do
    # 定义并处理每个训练数据集
    # for TASK in "AQuA" "ASDiv" "ASDiv_Grade_1" "ASDiv_Grade_2" "ASDiv_Grade_3" "ASDiv_Grade_4" "ASDiv_Grade_5" "ASDiv_Grade_6" "GSM" "LeeTCode_submission" "MultiArith" "SVAMP" "olympic_OE_TO_maths_en_COMP" "olympic_OE_TO_physics_en_COMP" "olympic_TP_TO_maths_en_COMP" "olympic_TP_TO_physics_en_COMP"
    for TASK in "ASDiv" "GSM" "LeeTCode" "LeeTCode_code_high" "LeeTCode_code_medium" "LeeTCode_code_low" "MultiArith" "SVAMP" "olympic_OE_TO_maths_en_COMP" "olympic_OE_TO_physics_en_COMP" "olympic_TP_TO_maths_en_COMP" "olympic_TP_TO_physics_en_COMP"
    do
        MODEL_PATH=../out/${MODEL_NAME}-p0.05-lora-seed${SEED}/checkpoint-${CKPT}
        OUTPUT_PATH=../grads/${MODEL_NAME}-p0.05-lora-seed${SEED}/${TASK}-ckpt${CKPT}-sgd
        
        # 执行数据处理脚本
        echo "Processing $TASK at checkpoint $CKPT..."
        ./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"

        # 打印状态消息
        echo "Processed testing data for CKPT=$CKPT with TASK=$TASK"
    done
done