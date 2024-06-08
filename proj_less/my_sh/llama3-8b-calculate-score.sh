# !运行这一段代码前需要注意看看学习率的设置


pwd
chmod +x ./less/scripts/data_selection/matching.sh 
export CUDA_VISIBLE_DEVICES=1


DIM=8192
SEED=3
MODEL_NAME="llama3-8b"
SELECTED_DATA_OUTPUT_PATH=../selected_data/${MODEL_NAME}
TRAIN_FILE_NAMES="code_high code_medium code_low"


CKPTS="2 4 6 8"
# CHECKPOINT_WEIGHTS="1e-06 1e-06 1e-06 1e-06"
CHECKPOINT_WEIGHTS="1.7142857142857142e-05 1.1428571428571429e-05 5.7142857142857145e-06 2.8571428571428573e-06" 





GRADIENT_PATH=../grads/${MODEL_NAME}-p0.05-lora-seed${SEED}/{}-ckpt{}-adam/dim${DIM}/all_orig.pt
VALIDATION_GRADIENT_PATH=../grads/${MODEL_NAME}-p0.05-lora-seed${SEED}/{}-ckpt{}-sgd/dim${DIM}/all_orig.pt


TARGET_TASK_NAMES="AQuA ASDiv ASDiv_Grade_1 ASDiv_Grade_2 ASDiv_Grade_3 ASDiv_Grade_4 ASDiv_Grade_5 ASDiv_Grade_6 GSM LeeTCode_submission MultiArith SVAMP olympic_OE_TO_maths_en_COMP olympic_OE_TO_physics_en_COMP olympic_TP_TO_maths_en_COMP olympic_TP_TO_physics_en_COMP"

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"
