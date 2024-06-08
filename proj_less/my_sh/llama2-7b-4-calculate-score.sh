# !运行这一段代码前需要注意看看学习率的设置


pwd
chmod +x ./less/scripts/data_selection/matching.sh 
export CUDA_VISIBLE_DEVICES=0


DIM=8192
SEED=3
MODEL_NAME="llama2-7b"
SELECTED_DATA_OUTPUT_PATH=../selected_data/${MODEL_NAME}
TRAIN_FILE_NAMES="code_high code_medium code_low"


CKPTS="4 8 12 16"
CHECKPOINT_WEIGHTS="1e-06 1e-06 1e-06 1e-06"
CHECKPOINT_WEIGHTS="1.6000000000000003e-05 1.0666666666666667e-05 5.333333333333334e-06 1.3333333333333334e-06" 


# 梯度位置，
GRADIENT_PATH=../grads/${MODEL_NAME}-p0.05-lora-seed${SEED}/{}-ckpt{}-adam/dim${DIM}/all_orig.pt
VALIDATION_GRADIENT_PATH=../grads/${MODEL_NAME}-p0.05-lora-seed${SEED}/{}-ckpt{}-sgd/dim${DIM}/all_orig.pt


#TARGET_TASK_NAMES="AQuA ASDiv ASDiv_Grade_1 ASDiv_Grade_2 ASDiv_Grade_3 ASDiv_Grade_4 ASDiv_Grade_5 ASDiv_Grade_6 GSM LeeTCode_submission MultiArith SVAMP olympic_OE_TO_maths_en_COMP olympic_OE_TO_physics_en_COMP olympic_TP_TO_maths_en_COMP olympic_TP_TO_physics_en_COMP"
TARGET_TASK_NAMES="LeeTCode_code_high LeeTCode_code_medium LeeTCode_code_low"

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"
