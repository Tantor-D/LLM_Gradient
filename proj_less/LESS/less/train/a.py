import os

for root, dirs, files in os.walk("../out/llama2-7b-p0.05-lora-seed3"):
    for file in files:
        if file == "pytorch_model_fsdp.bin":
            file_path = os.path.join(root, file)
            print(f"Deleting: {file_path}")  # 打印将要删除的文件路径
            os.remove(file_path)  # 删除文件