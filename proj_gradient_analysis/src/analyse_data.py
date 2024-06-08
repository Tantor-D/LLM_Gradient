# todo 之后leecode需要把数据按照难度分个类再做一遍（不同人打的难度评分不一样）
# todo 看看leetcode的数据集跑一下CIRS的结果，之后看看把数据集分成几个类别，然后看看每个类别的数据集的相似度[这部分其实不需要显卡]
# todo 需要尝试聚类
# todo 聚类做不到的话，可以尝试先找几个测试数据集，跑一遍CIRS，然后看看相似度的情况
import json
import os
from collections import Counter

from matplotlib import pyplot as plt


# 分析LESS得到的结果输出图和csv
# 每个测试集，分析相似度前5%和1%的训练数据占比
# 每个训练数据，分析在测试集上5%和1% 出现的数据频次，绘制柱状图。 然后是所属的数据集出现的频次


def analyse_single_task(file_path):
    """读入一个jsonl文件"""
    train_data_id_count_5 = {}  # 5%数据，统计训练数据出现的频次
    train_type_count_5 = {}  # 5%数据，统计类别出现的频次
    train_data_id_count_1 = {}  # 1%数据，统计训练数据出现的频次
    train_type_count_1 = {}  # 1%数据，统计类别出现的频次

    total_line = 0
    with open(file_path, 'r') as file:
        for line in file:
            total_line += 1

    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            json_data = json.loads(line)
            train_data_id = json_data["id"]
            dataset_name = json_data["dataset"]

            train_data_id_count_5[train_data_id] = train_data_id_count_5.get(train_data_id, 0) + 1
            train_type_count_5[dataset_name] = train_type_count_5.get(dataset_name, 0) + 1
            if idx < total_line / 5:
                train_data_id_count_1[train_data_id] = train_data_id_count_1.get(train_data_id, 0) + 1
                train_type_count_1[dataset_name] = train_type_count_1.get(dataset_name, 0) + 1

    return train_data_id_count_5, train_type_count_5, train_data_id_count_1, train_type_count_1


def plot_top_items(data, x, save_pic_path, title):
    # 使用Counter来处理数据并选出前x个最常见的元素
    top_items = Counter(data).most_common(x)

    # 解包top_items以便绘图
    names, frequencies = zip(*top_items)
    names = [name.replace("code_high", "high") for name in names]
    names = [name.replace("code_medium", "m") for name in names]
    names = [name.replace("code_low", "low") for name in names]

    # 创建柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(names, frequencies, color='blue')
    plt.xlabel('Items')
    plt.ylabel('Frequency')
    plt.title(title)

    plt.xticks(rotation=45, fontsize=10)  # 旋转标签和调整字体大小
    plt.tight_layout()  # 自动调整子图参数，以给定的填充方式

    plt.savefig(save_pic_path)
    plt.close()


def draw_pic(data_dict, title, save_path):
    # 画扇形图
    labels = ["code_medium", "code_high", "code_low"]
    values = []
    for key in labels:
        values.append(data_dict.get(key, 0))

    fig, ax = plt.subplots()
    # 绘制饼状图
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    # 添加标题 + 图例
    ax.set_title(title)
    ax.legend(labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    # 保存图形为PNG文件，文件名基于索引编号
    plt.savefig(save_path, bbox_inches='tight')  # bbox_inches='tight'用于确保图例也被包含在图内
    plt.close(fig)  # 关闭图形以释放内存


if __name__ == "__main__":
    train_dataset_name = ["code_low", "code_medium", "code_high"]
    eval_dataset_name = ["AQuA", "ASDiv", "ASDiv_Grade_1", "ASDiv_Grade_2", "ASDiv_Grade_3", "ASDiv_Grade_4",
                         "ASDiv_Grade_5", "ASDiv_Grade_6", "GSM", "LeeTCode_submission", "MultiArith", "SVAMP",
                         "olympic_OE_TO_maths_en_COMP", "olympic_OE_TO_physics_en_COMP",
                         "olympic_TP_TO_maths_en_COMP", "olympic_TP_TO_physics_en_COMP",
                         ]
    llm_model_name_list = ["llama2-7b", "llama3-8b"]

    config_list = [
        {
            "llm_name": "llama2-7b",
            "train_dataset_remark": "训练数据数量上是一类870条数据",
            "selected_data_path": r"E:\backup_for_servicer\1_project\project_870_code\selected_data",
            "save_path": "./analyse_result/llama2-7b-870"
        },
        {
            "llm_name": "llama3-8b",
            "train_dataset_remark": "训练数据数量上是一类870条数据",
            "selected_data_path": r"E:\backup_for_servicer\1_project\project_870_code\selected_data",
            "save_path": "./analyse_result/llama3-8b-870"
        },
        {
            "llm_name": "llama2-7b",
            "train_dataset_remark": "训练数据数量上是一类1700条数据",
            "selected_data_path": r"E:\backup_for_servicer\1_project\project_1700_code\selected_data",
            "save_path": "./analyse_result/llama2-7b-1700"
        }
    ]

    result = []
    # 首先构建出json分析数据
    for config in config_list:
        cur_result = {"config": config}
        total_train_data_id_count_5 = {}
        total_train_data_id_count_1 = {}
        for eval_dataset in eval_dataset_name:
            train_data_id_count_5, train_type_count_5, train_data_id_count_1, train_type_count_1 = analyse_single_task(
                f"{config['selected_data_path']}/{config['llm_name']}/{eval_dataset}/top_p0.05.jsonl")
            for train_data_id, count in train_data_id_count_5.items():
                total_train_data_id_count_5[train_data_id] = total_train_data_id_count_5.get(train_data_id, 0) + count
            for train_data_id, count in train_data_id_count_1.items():
                total_train_data_id_count_1[train_data_id] = total_train_data_id_count_1.get(train_data_id, 0) + count
            cur_result[f"{eval_dataset}"] = {
                "train_type_count_5": train_type_count_5,
                "train_type_count_1": train_type_count_1,
                "train_data_id_count_5": train_data_id_count_5,
                "train_data_id_count_1": train_data_id_count_1
            }
        cur_result["total_train_data_id_count_5"] = total_train_data_id_count_5
        cur_result["total_train_data_id_count_1"] = total_train_data_id_count_1
        result.append(cur_result)

    analyse_result_path = "./analyse_result"
    os.makedirs(analyse_result_path, exist_ok=True)
    with open(f"{analyse_result_path}/result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    # 根据json数据绘制出饼状图和柱状图，按照每个的config来处理
    for data in result:
        this_save_path = data["config"]["save_path"]
        os.makedirs(this_save_path, exist_ok=True)
        llm_name = data["config"]["llm_name"]
        train_dataset_remark = data["config"]["train_dataset_remark"]
        for key, val in data.items():
            if key == "config":
                continue
            if key == "total_train_data_id_count_5":
                plot_top_items(val, 30, f"{this_save_path}/0_frequency_{llm_name}_top0.05.png",
                               f"{llm_name}_top0.05")
                continue
            if key == "total_train_data_id_count_1":
                plot_top_items(val, 30, f"{this_save_path}/0_frequency_{llm_name}_top0.01.png",
                               f"{llm_name}_top0.01")
                continue

            # key此时是eval datasetName
            save_path_5 = f"{this_save_path}/{llm_name}_{key}_top0.05.png"
            save_path_1 = f"{this_save_path}/{llm_name}_{key}_top0.01.png"
            draw_pic(val["train_type_count_5"], f"{llm_name}_{key}_top0.05", save_path_5)
            draw_pic(val["train_type_count_1"], f"{llm_name}_{key}_top0.01", save_path_1)

            # 绘制柱状图
            pass
