"""将CIRS项目中给出的代码数据转化为对话形式的数据"""

import json
import os


def reformat(input_path, output_path, dataset_name, prompt_kind="alpaca"):
    # 读取 JSON 文件
    with open(input_path, 'r') as file:
        data_list = json.load(file)

    # 写入 JSONL 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for idx, data in enumerate(data_list):
            # 构造 prompt，确保以换行符结束
            if prompt_kind == "alpaca":
                # 按照alpaca那篇文章的prompt构建方法来做的
                instruction = data['instruction']
                input_str = data['input']
                if input_str:
                    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:"
                else:
                    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"""
            else:
                prompt = data['instruction']
                if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                    prompt += "\n"
                if data['input']:
                    prompt += "Input data is " + data['input'] + "\n"

            response = data['output']
        
            # 将数据以 JSONL 格式写入文件
            json_object = json.dumps({
                "dataset": dataset_name,
                "id": f"{dataset_name}_{idx}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            })
            f.write(json_object + "\n")


if __name__ == "__main__":
    config_list = [
        # {
        #     "input_path": "/mnt/zhiyuan/dsw/proj_easy/EasyInstruct/out_data/selections/manually_pration_results/data_cleaned_high.json",
        #     "output_path": "./example_code/code_high/code_high_data.jsonl",
        #     "dataset_name": "code_high"
        # },
        # {
        #     "input_path": "/mnt/zhiyuan/dsw/proj_easy/EasyInstruct/out_data/selections/manually_pration_results/data_cleaned_low.json",
        #     "output_path": "./example_code/code_low/code_low_data.jsonl",
        #     "dataset_name": "code_low"
        # },
        # {
        #     "input_path": "/mnt/zhiyuan/dsw/proj_easy/EasyInstruct/out_data/selections/manually_pration_results/data_cleaned_medium.json",
        #     "output_path": "./example_code/code_medium/code_medium_data.jsonl",
        #     "dataset_name": "code_medium"
        # }

        {
            "input_path": "/mnt/zhiyuan/dsw/proj_easy/EasyInstruct/out_data/selections/manually_pration_results/data_cleaned_high.json",
            "output_path": "./example_code_alpaca/code_high/code_high_data.jsonl",
            "dataset_name": "code_high",
            "kind": "alpaca"
        },
        {
            "input_path": "/mnt/zhiyuan/dsw/proj_easy/EasyInstruct/out_data/selections/manually_pration_results/data_cleaned_low.json",
            "output_path": "./example_code_alpaca/code_low/code_low_data.jsonl",
            "dataset_name": "code_low",
            "kind": "alpaca"
        },
        {
            "input_path": "/mnt/zhiyuan/dsw/proj_easy/EasyInstruct/out_data/selections/manually_pration_results/data_cleaned_medium.json",
            "output_path": "./example_code_alpaca/code_medium/code_medium_data.jsonl",
            "dataset_name": "code_medium",
            "kind": "alpaca"
        }
    ]
    for config in config_list:
        reformat(config["input_path"], config["output_path"], config["dataset_name"], config.get("kind", ""))
