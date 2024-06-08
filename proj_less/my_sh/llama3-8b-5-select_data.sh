MODEL_NAME="llama3-8b"
OUTPU_BASE_DIR=../selected_data/${MODEL_NAME}


TARGET_TASK_NAMES="AQuA ASDiv ASDiv_Grade_1 ASDiv_Grade_2 ASDiv_Grade_3 ASDiv_Grade_4 ASDiv_Grade_5 ASDiv_Grade_6 GSM LeeTCode_submission MultiArith SVAMP olympic_OE_TO_maths_en_COMP olympic_OE_TO_physics_en_COMP olympic_TP_TO_maths_en_COMP olympic_TP_TO_physics_en_COMP"
TRAIN_FILE_NAMES="code_high code_medium code_low"
TRAIN_FILES="../data/train/processed/code_high/code_high_data.jsonl ../data/train/processed/code_medium/code_medium_data.jsonl ../data/train/processed/code_low/code_low_data.jsonl "


python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--percentage 0.05 \
--train_files ${TRAIN_FILES[@]} \
--output_path ${OUTPU_BASE_DIR}