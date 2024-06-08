import os
import subprocess

# Set environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define variables
dim = 8192
seed = 3
model_name = "llama2-7b"
selected_data_output_path = f'../selected_data/{model_name}'
train_file_names = "code_high code_medium code_low"

ckpts = "4 8 12 16"
checkpoint_weights = "1.6000000000000003e-05 1.0666666666666667e-05 5.333333333333334e-06 1.3333333333333334e-06"
gradient_path = f'../grads/{model_name}-p0.05-lora-seed{seed}/{{}}-ckpt{{}}-adam/dim{dim}/all_orig.pt'
validation_gradient_path = f'../grads/{model_name}-p0.05-lora-seed{seed}/{{}}-ckpt{{}}-sgd/dim{dim}/all_orig.pt'
target_task_names = "LeeTCode_code_high LeeTCode_code_medium LeeTCode_code_low"

# Define the matching script path
matching_script_path = "./less/scripts/data_selection/matching.sh"

# Check if output directory exists, if not, create it
if not os.path.exists(selected_data_output_path):
    os.makedirs(selected_data_output_path)

# Build the command to execute
command = [
    matching_script_path,
    gradient_path,
    train_file_names,
    ckpts,
    checkpoint_weights,
    validation_gradient_path,
    target_task_names,
    selected_data_output_path
]

# Execute the script
subprocess.run(command, shell=True)
