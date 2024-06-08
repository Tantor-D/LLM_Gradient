import json
import os
from collections import Counter

from matplotlib import pyplot as plt


def plot_top_items(data, x, title, save_pic_path):
    # 输入的data是一个字典
    # 使用Counter来处理数据并选出前x个最常见的元素
    top_items = Counter(data).most_common(x)
    # top_items = sorted(data.items(), key=lambda item: item[1], reverse=True)[:x]

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
    valid_dataset_name = ["AQuA", "ASDiv", "ASDiv_Grade_1", "ASDiv_Grade_2", "ASDiv_Grade_3", "ASDiv_Grade_4",
                          "ASDiv_Grade_5", "ASDiv_Grade_6", "GSM", "LeeTCode_submission", "MultiArith", "SVAMP",
                          "olympic_OE_TO_maths_en_COMP", "olympic_OE_TO_physics_en_COMP",
                          "olympic_TP_TO_maths_en_COMP", "olympic_TP_TO_physics_en_COMP"]
    config_list = [
        {
            "code_num": 870,
            "model_name": "llama2-7b",
            "sample_sizes": [5, 10, 50],
            "base_data_path": r"E:\backup_for_servicer\1_project\analyse_result",
            "save_base_path": r"E:\backup_for_servicer\1_project\analyse_result\pics"
        },
        {
            "code_num": 870,
            "model_name": "llama3-8b",
            "sample_sizes": [5, 10, 50],
            "base_data_path": r"E:\backup_for_servicer\1_project\analyse_result",
            "save_base_path": r"E:\backup_for_servicer\1_project\analyse_result\pics"
        },
        {
            "code_num": 1700,
            "model_name": "llama2-7b",
            "sample_sizes": [5, 10, 50],
            "base_data_path": r"E:\backup_for_servicer\1_project\analyse_result",
            "save_base_path": r"E:\backup_for_servicer\1_project\analyse_result\pics"
        }
    ]

    for config in config_list:
        for sample_size in config["sample_sizes"]:
            for dataset_name in valid_dataset_name:
                code_num = config["code_num"]
                model_name = config["model_name"]
                id_count_path = os.path.join(config["base_data_path"], f"{code_num}_code", f"{sample_size}",
                                             f"{model_name}", f"{dataset_name}_data_id_count.json")
                dataset_count_path = os.path.join(config["base_data_path"], f"{code_num}_code", f"{sample_size}",
                                                  f"{model_name}", f"{dataset_name}_dataset_count.json")

                # 确定好保存的路径
                base_save_path = os.path.join(config["save_base_path"], f"{code_num}_code", f"{model_name}")
                top5_save_path = os.path.join(base_save_path, "top5")
                top1_save_path = os.path.join(base_save_path, "top1")
                os.makedirs(top5_save_path, exist_ok=True)
                os.makedirs(top1_save_path, exist_ok=True)

                # 读取数据准备绘制
                with open(id_count_path, "r") as f:
                    id_count_data = json.load(f)
                    plot_top_items(id_count_data["top5"], 10, f"{dataset_name}_sample_{sample_size}_{model_name}",
                                   save_pic_path=os.path.join(top5_save_path,
                                                              f"{dataset_name}_sample_{sample_size}_id_count.png"))
                    plot_top_items(id_count_data["top1"], 10, f"{dataset_name}_sample_{sample_size}_{model_name}",
                                   save_pic_path=os.path.join(top1_save_path,
                                                              f"{dataset_name}_sample_{sample_size}_id_count.png"))

                with open(dataset_count_path, "r") as f:
                    dataset_count_data = json.load(f)
                    draw_pic(dataset_count_data["top5"], f"{dataset_name}_sample_{sample_size}_{model_name}",
                             os.path.join(top5_save_path, f"{dataset_name}_sample_{sample_size}_dataset.png"))
                    draw_pic(dataset_count_data["top1"], f"{dataset_name}_sample_{sample_size}_{model_name}",
                             os.path.join(top1_save_path, f"{dataset_name}_sample_{sample_size}_dataset.png"))
