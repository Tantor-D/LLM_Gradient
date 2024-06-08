import json
import os

import torch


def deal_tensor(CKPT, classifcation, ori_data, source_torch_files_path, torch_target_files_path):
    tensors = {
        "code_high": None,
        "code_medium": None,
        "code_low": None
    }
    for ckpt in CKPT:
        source_path = source_torch_files_path.format(ckpt)
        # 加载torch文件
        torch_data = torch.load(source_path)
        for idx, d in enumerate(ori_data):
            d = json.loads(d)
            code_type = classifcation.get(d["id"], None)
            if not code_type:
                continue
            row = torch_data[idx]
            tensors[code_type] = torch.cat((tensors[code_type], row.unsqueeze(0)), dim=0) if tensors[code_type] is not None else row.unsqueeze(0)

        # 保存torch文件
        for k, v in torch_target_files_path.items():
            target_path = v.format(ckpt)
            os.makedirs(os.path.dirname(target_path)) if not os.path.exists(os.path.dirname(target_path)) else None
            torch.save(tensors[k], target_path)


if __name__ == "__main__":

    # 获取分类结果
    classifcation = {}
    with open(r"E:\backup_for_servicer\1_project\my\leetcode_cirs\data_cleaned_high.json", "r") as f:
        data = json.load(f)
        for d in data:
            classifcation[d["id"]] = "code_high"

    with open(r"E:\backup_for_servicer\1_project\my\leetcode_cirs\data_cleaned_medium.json", "r") as f:
        data = json.load(f)
        for d in data:
            classifcation[d["id"]] = "code_medium"

    with open(r"E:\backup_for_servicer\1_project\my\leetcode_cirs\data_cleaned_low.json", "r") as f:
        data = json.load(f)
        for d in data:
            classifcation[d["id"]] = "code_low"

    # print(classifcation)
    print(len(classifcation))

    # 将分类结果写入到LeeTCode_submission_test_1.jsonl
    with open(r"E:\backup_for_servicer\1_project\project_1700_code\data\eval\LeeTCode_submission_test.jsonl", "r") as f:
        data = f.readlines()
        new_data = []
        for d in data:
            new_d = {}
            d = json.loads(d)
            if d["id"] in classifcation.keys():
                if classifcation[d["id"]] not in ["code_high", "code_medium", "code_low"]:
                    print("xxxxxxxxxxxxx")
                new_d["dataset"] = d["dataset"]
                new_d["id"] = d["id"]
                new_d["label"] = classifcation[d["id"]]
                new_d["messages"] = d["messages"]
                new_data.append(new_d)

    with open(r"E:\backup_for_servicer\1_project\project_1700_code\data\eval\LeeTCode_submission_test_1.jsonl",
              "w") as f:
        for d in new_data:
            f.write(json.dumps(d) + "\n")

    # 构建出测试集，并同时构建出不同的梯度文件，同步构建出不同的torch文件
    target_files_path = {
        "code_high": r"E:\backup_for_servicer\1_project\project_1700_code\data\eval\LeeTCode_code_high_test.jsonl",
        "code_medium": r"E:\backup_for_servicer\1_project\project_1700_code\data\eval\LeeTCode_code_medium_test.jsonl",
        "code_low": r"E:\backup_for_servicer\1_project\project_1700_code\data\eval\LeeTCode_code_low_test.jsonl",
    }
    new_code_data = {
        "code_high": [],
        "code_medium": [],
        "code_low": [],
    }
    with open(r"E:\backup_for_servicer\1_project\project_1700_code\data\eval\LeeTCode_submission_test_1.jsonl",
              "r") as f:
        data = f.readlines()
    new_data = []
    for d in data:
        d = json.loads(d)
        new_code_data[d["label"]].append(d)

    # 分类保存原始数据
    for k, v in new_code_data.items():
        with open(target_files_path[k], "w") as f:
            for d in v:
                f.write(json.dumps(d) + "\n")

    with open(r"E:\backup_for_servicer\1_project\project_1700_code\data\eval\LeeTCode_submission_test.jsonl", "r") as f:
        ori_data = f.readlines()

    # 分类保存torch数据，870条数据的部分
    source_torch_files_path = r"E:\backup_for_servicer\1_project\project_870_code\grads\llama2-7b-p0.05-lora-seed3\LeeTCode_submission-ckpt{}-sgd\dim8192\all_orig.pt"
    torch_target_files_path = {
        "code_high": r"E:\backup_for_servicer\1_project\project_870_code\grads\llama2-7b-p0.05-lora-seed3\LeeTCode_code_high-ckpt{}-sgd\dim8192\all_orig.pt",
        "code_medium": r"E:\backup_for_servicer\1_project\project_870_code\grads\llama2-7b-p0.05-lora-seed3\LeeTCode_code_medium-ckpt{}-sgd\dim8192\all_orig.pt",
        "code_low": r"E:\backup_for_servicer\1_project\project_870_code\grads\llama2-7b-p0.05-lora-seed3\LeeTCode_code_low-ckpt{}-sgd\dim8192\all_orig.pt",
    }
    CKPT = [4, 8, 12, 16]
    deal_tensor(CKPT, classifcation, ori_data, source_torch_files_path, torch_target_files_path)

    source_torch_files_path = r"E:\backup_for_servicer\1_project\project_870_code\grads\llama3-8b-p0.05-lora-seed3\LeeTCode_submission-ckpt{}-sgd\dim8192\all_orig.pt"
    torch_target_files_path = {
        "code_high": r"E:\backup_for_servicer\1_project\project_870_code\grads\llama3-8b-p0.05-lora-seed3\LeeTCode_code_high-ckpt{}-sgd\dim8192\all_orig.pt",
        "code_medium": r"E:\backup_for_servicer\1_project\project_870_code\grads\llama3-8b-p0.05-lora-seed3\LeeTCode_code_medium-ckpt{}-sgd\dim8192\all_orig.pt",
        "code_low": r"E:\backup_for_servicer\1_project\project_870_code\grads\llama3-8b-p0.05-lora-seed3\LeeTCode_code_low-ckpt{}-sgd\dim8192\all_orig.pt",
    }
    CKPT = [2, 4, 6, 8]
    deal_tensor(CKPT, classifcation, ori_data, source_torch_files_path, torch_target_files_path)

    # 分类保存torch数据，1700条数据的部分
    source_torch_files_path = r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\LeeTCode_submission-ckpt{}-sgd\dim8192\all_orig.pt"
    torch_target_files_path = {
        "code_high": r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\LeeTCode_code_high-ckpt{}-sgd\dim8192\all_orig.pt",
        "code_medium": r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\LeeTCode_code_medium-ckpt{}-sgd\dim8192\all_orig.pt",
        "code_low": r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\LeeTCode_code_low-ckpt{}-sgd\dim8192\all_orig.pt",
    }
    CKPT = [7, 15, 23, 28]
    deal_tensor(CKPT, classifcation, ori_data, source_torch_files_path, torch_target_files_path)
