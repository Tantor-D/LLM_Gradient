import copy
import json
import os
from collections import Counter

from matplotlib import pyplot as plt



if __name__ == "__main__":
    # 计算相同数据集不同模型的差值
    file = r"E:\backup_for_servicer\1_project\analyse_result\difference\difference between llama2-7b_data870 and llama3-8b_data870_average_final.json"
    with open(file, "r") as f:
        all_data = json.load(f)
    data_count = {"code_high": [], "code_medium": [], "code_low": []}
    for dataset in all_data:
        if dataset == "info":
            continue
        for top_kind in ["top1", "top5"]:
            for code_kind in ["code_high", "code_medium", "code_low"]:
                data_count[code_kind].append(all_data[dataset][top_kind][code_kind])
    ave_data = {"code_high": sum(data_count["code_high"]) / len(data_count["code_high"]),
                "code_medium": sum(data_count["code_medium"]) / len(data_count["code_medium"]),
                "code_low": sum(data_count["code_low"]) / len(data_count["code_low"])}
    print(ave_data)


if __name__ == "__main__":
    valid_dataset_name = ["AQuA", "ASDiv", "ASDiv_Grade_1", "ASDiv_Grade_2", "ASDiv_Grade_3", "ASDiv_Grade_4",
                          "ASDiv_Grade_5", "ASDiv_Grade_6", "GSM", "LeeTCode_submission", "MultiArith", "SVAMP",
                          "olympic_OE_TO_maths_en_COMP", "olympic_OE_TO_physics_en_COMP",
                          "olympic_TP_TO_maths_en_COMP", "olympic_TP_TO_physics_en_COMP"]

    base_config = {
        "sample_sizes": [5, 10, 50],
        "base_data_path": r"E:\backup_for_servicer\1_project\analyse_result",
        "top_size": [1, 5],
        "base_dir": r"E:\backup_for_servicer\1_project\analyse_result",
    }
    config_list = [
        # {
        #     "code_num": 870,
        #     "model_name": "llama2-7b",
        # },
        {
            "code_num": 870,
            "model_name": "llama3-8b",
        },
        {
            "code_num": 1700,
            "model_name": "llama2-7b",
        }
    ]
    for config in config_list:
        config.update(base_config)

    diff_dict = {
        "info": f"difference between {config_list[0]['model_name']}_data{config_list[0]['code_num']} and {config_list[1]['model_name']}_data{config_list[1]['code_num']}"}
    for dataset in valid_dataset_name:
        diff_dict[dataset] = {"top1": {"code_high": [], "code_medium": [], "code_low": []},
                              "top5": {"code_high": [], "code_medium": [], "code_low": []}}

    # 首先计算完具体的小dataset的所有差值信息
    code_kind_list = ["code_high", "code_medium", "code_low"]
    top_kind_list = ["top1", "top5"]
    for dataset in valid_dataset_name:
        for code_kind in code_kind_list:
            for sample_size in config_list[0]["sample_sizes"]:
                with open(os.path.join(config_list[0]["base_data_path"],
                                       f"{config_list[0]['code_num']}_code",
                                       str(sample_size),
                                       f"{config_list[0]['model_name']}",
                                       f"{dataset}_dataset_count.json"), "r") as f:
                    data1 = json.load(f)
                    sum_top1 = sum(data1["top1"].values())
                    sum_top5 = sum(data1["top5"].values())
                    for code_kinddd in code_kind_list:
                        data1["top1"][code_kinddd] = data1["top1"].get(code_kinddd, 0) / sum_top1
                        data1["top5"][code_kinddd] = data1["top5"].get(code_kinddd, 0) / sum_top5
                with open(os.path.join(config_list[1]["base_data_path"],
                                       f"{config_list[1]['code_num']}_code",
                                       str(sample_size), f"{config_list[1]['model_name']}",
                                       f"{dataset}_dataset_count.json"), "r") as f:
                    data2 = json.load(f)
                    sum_top1 = sum(data2["top1"].values())
                    sum_top5 = sum(data2["top5"].values())
                    for code_kinddd in code_kind_list:
                        data2["top1"][code_kinddd] = data2["top1"].get(code_kinddd, 0) / sum_top1
                        data2["top5"][code_kinddd] = data2["top5"].get(code_kinddd, 0) / sum_top5
                diff_dict[dataset]["top1"][code_kind].append(data1["top1"][code_kind] - data2["top1"][code_kind])
                diff_dict[dataset]["top5"][code_kind].append(data1["top5"][code_kind] - data2["top5"][code_kind])

                os.makedirs(os.path.join(config_list[0]["base_dir"], "difference"), exist_ok=True)
                with open(os.path.join(config_list[0]["base_data_path"],
                                       "difference",
                                       f"{diff_dict['info']}.json"), "w") as f:
                    json.dump(diff_dict, f, indent=4)

    # 获取每个小数据集的平均差值信息
    average_diff_dict = copy.deepcopy(diff_dict)
    average_diff_dict["info"] = average_diff_dict["info"] + "_average"
    for dataset in average_diff_dict:
        if dataset == "info":
            continue
        for top_kind in top_kind_list:
            for code_kind in code_kind_list:
                summ = 0
                lenn = 0
                for data in diff_dict[dataset][top_kind][code_kind]:
                    summ += abs(data)
                    lenn += 1
                average_diff_dict[dataset][top_kind][code_kind] = summ / lenn
    with open(os.path.join(config_list[0]["base_data_path"],
                           "difference",
                           f"{average_diff_dict['info']}.json"), "w") as f:
        json.dump(average_diff_dict, f, indent=4)

    # 计算最终的大dataset的差值信息
    final_diff_dict = copy.deepcopy(average_diff_dict)
    asdiv_diff_dict = {}
    olympic_diff_dict = {}
    pop_list = []
    for dataset in final_diff_dict:
        if dataset == "info":
            final_diff_dict["info"] = final_diff_dict["info"] + "_final"
            continue
        if dataset.startswith("ASDiv") and dataset != "ASDiv":
            asdiv_diff_dict[dataset] = final_diff_dict[dataset]
            pop_list.append(dataset)
            continue
        if dataset.startswith("olympic"):
            olympic_diff_dict[dataset] = final_diff_dict[dataset]
            pop_list.append(dataset)
            continue
    for dataset in pop_list:
        final_diff_dict.pop(dataset)

    # 处理这俩数据集
    asdiv_final = {"top1": {"code_high": [], "code_medium": [], "code_low": []}, "top5": {"code_high": [], "code_medium": [], "code_low": []}}
    olympic_final = {"top1": {"code_high": [], "code_medium": [], "code_low": []}, "top5": {"code_high": [], "code_medium": [], "code_low": []}}
    for dataset in asdiv_diff_dict:
        for top_kind in top_kind_list:
            for code_kind in code_kind_list:
                asdiv_final[top_kind][code_kind].append(asdiv_diff_dict[dataset][top_kind][code_kind])
    for dataset in olympic_diff_dict:
        for top_kind in top_kind_list:
            for code_kind in code_kind_list:
                olympic_final[top_kind][code_kind].append(olympic_diff_dict[dataset][top_kind][code_kind])
    for top_kind in top_kind_list:
        for code_kinddd in ["code_high", "code_medium", "code_low"]:
            asdiv_final[top_kind][code_kinddd] = sum(asdiv_final[top_kind][code_kinddd]) / len(asdiv_final[top_kind][code_kinddd])
            olympic_final[top_kind][code_kinddd] = sum(olympic_final[top_kind][code_kinddd]) / len(olympic_final[top_kind][code_kinddd])
    final_diff_dict["ASDiv_average"] = asdiv_final
    final_diff_dict["olympic_average"] = olympic_final
    with open(os.path.join(config_list[0]["base_data_path"],
                           "difference",
                           f"{final_diff_dict['info']}.json"), "w") as f:
        json.dump(final_diff_dict, f, indent=4)



