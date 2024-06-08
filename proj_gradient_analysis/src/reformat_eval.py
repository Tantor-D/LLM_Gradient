import os
import json


def reformatGSM(input_path, output_path, dataset_name):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_data_list = []
    # 打开文件
    with open(input_path, 'r') as file:
        for idx, line in enumerate(file):
            data_dict = json.loads(line.strip())

            processed_data = {
                "dataset": dataset_name,
                "id": f"{dataset_name}_{idx}",
                "messages": [
                    {"role": "user", "content": data_dict["question"]},
                    {"role": "assistant", "content": data_dict["answer"]}
                ]
            }

            json_object = json.dumps(processed_data)
            final_data_list.append(json_object)
    
    with open(output_path, 'w') as f:
        for final_data in final_data_list:
            f.write(final_data + "\n")



def reformatAQuA(input_path, output_path, dataset_name):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_data_list = []
    # 打开文件
    with open(input_path, 'r') as file:
        for idx, line in enumerate(file):
            data_dict = json.loads(line.strip())

            # 用的alpaca的prompt形式
            
            option_str = ", ".join(data_dict["options"])
            prompt = f"Below is an instruction that describes a task, paired with some options. Please analyze the options carefully and select the most appropriate one. Write a response that appropriately completes the request.\n\n### Instruction:\n{data_dict['question']}\n### Options:\n{option_str}\n### Response:"


            processed_data = {
                "dataset": dataset_name,
                "id": f"{dataset_name}_{idx}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": data_dict["rationale"]}
                ]
            }

            json_object = json.dumps(processed_data)
            final_data_list.append(json_object)
    
    with open(output_path, 'w') as f:
        for final_data in final_data_list:
            f.write(final_data + "\n")


if __name__ == "__main__":
    config_list = [
        {
            "input_path": "/mnt/zhiyuan/dsw/raw_dataset/AQuA/test.json",
            "output_path": "./formated_eval/AQuA_test.jsonl",
            "dataset_name": "AQuA"
        },
        {
            "input_path": "/mnt/zhiyuan/dsw/raw_dataset/grade-school-math/grade_school_math/data/test.jsonl",
            "output_path": "./formated_eval/GSM_test.jsonl",
            "dataset_name": "GSM"
        }
    ]

    for config in config_list:
        if "AQuA" in config["dataset_name"]:
            reformatAQuA(config["input_path"], config["output_path"], config["dataset_name"])
        elif "GSM" in config["dataset_name"]:
            reformatGSM(config["input_path"], config["output_path"], config["dataset_name"])


