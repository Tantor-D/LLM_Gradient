import os
import json
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

class LayerAnalyser:
    """
    A class used to analyze layer data from JSON files and visualize it in Excel and bar charts.
    尽量写代码的时候都弄成 json -> excel -> pic。 毕竟excel和pic是最终的展示形式，都是需要的。

    Attributes:
        dataset_name_list (list[str]): A list of dataset names.
        model_name (str): The name of the model being analyzed.
        sample_size (int): The size of the sample data.
        base_data_dir (str): The base directory where data is stored.
        save_dir (str): The directory where results will be saved.
        from_layer_list (list[int]): List of starting layers.
        end_layer_list (list[int]): List of ending layers.
        dataset_data (dict): Data read from JSON files.
        percent_dataset_data (dict): Data converted to percentages.
    """

    def __init__(self,
                 dataset_name_list: list[str],
                 model_name: str,
                 sample_size: int,
                 base_data_dir: str,
                 save_dir: str,
                 ):
        """
        Initializes the LayerAnalyser with dataset names, model name, sample size, and directories.

        Args:
            dataset_name_list (list[str]): A list of dataset names.
            model_name (str): The name of the model being analyzed.
            sample_size (int): The size of the sample data.
            base_data_dir (str): The base directory where data is stored.
            save_dir (str): The directory where results will be saved.
        """
        self.dataset_name_list = dataset_name_list

        self.model_name = model_name
        assert model_name in ["llama2-7b"]

        self.base_data_dir: str = base_data_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.sample_size = sample_size

        # 需要后续进一步计算指定
        self.from_layer_list = [i for i in range(0, 32)]
        self.end_layer_list = [i for i in range(0, 32)]

        self.dataset_data = self.read_json_data(base_data_dir, dataset_name_list, sample_size)
        self.percent_dataset_data = self.trans_to_percent_data(self.dataset_data)

    def read_json_data(self, base_data_dir, dataset_name_list, sample_size: int):
        """
        Reads data from JSON files and returns it as a dictionary.
        暂时只会读取dataset_count文件，不会读data id的文件

        Args:
            base_data_dir (str): The base directory where data is stored.
            dataset_name_list (list[str]): A list of dataset names.
            sample_size (int): The size of the sample data.

        Returns:
            dict: A dictionary containing the data read from JSON files.
        """
        json_dir = os.path.join(base_data_dir, str(sample_size), self.model_name)

        # 确定要读取的数据集，单纯的使用_分割会出问题，因为数据集名字中可能会有_
        # if dataset_name_list is None or dataset_name_list == []:
        #     dataset_name_list = []
        #     # 遍历指定目录下的所有
        #     for filename in os.listdir(json_dir):
        #         # 获取文件名的第一部分（假设文件名中使用_进行分割）
        #         dataset_name = filename.split('_')[0]
        #         dataset_name_list.append(dataset_name) if dataset_name not in dataset_name_list else None
        print("in read_json_data, dataset_name_list: ", dataset_name_list)

        # ret_data的组织形式为 {dataset_name: {"layer": {"top5": count}}}
        ret_data = {}
        for dataset_name in dataset_name_list:
            ret_data[dataset_name] = {}
            for layer_idx in range(len(self.from_layer_list)):
                current_layer_str = f"layer_from_{self.from_layer_list[layer_idx]}_to_{self.end_layer_list[layer_idx]}"
                filename = f"{dataset_name}_dataset_count_{sample_size}_{current_layer_str}.json"
                dataset_path = os.path.join(json_dir, filename)
                if not os.path.exists(dataset_path):
                    print(f"file {dataset_path} not exists!!!!")
                    continue
                with open(dataset_path, "r") as f:
                    dataset_data = json.load(f)
                    ret_data[dataset_name][current_layer_str] = dataset_data

        return ret_data

    def trans_to_percent_data(self, data):
        """
        Converts raw data to percentage data.

        Args:
            data (dict): The raw data to be converted.

        Returns:
            dict: The data converted to percentages.
        """
        ret_data = copy.deepcopy(data)
        for dataset_name, dataset_data in ret_data.items():
            for layer, layer_data in dataset_data.items():
                for top_key, top_data in layer_data.items():
                    summ = 0
                    for data_key, data_value in top_data.items():
                        summ += data_value
                    for data_key, data_value in top_data.items():
                        ret_data[dataset_name][layer][top_key][data_key] = (data_value / summ) * 100
        return ret_data

    def trans_to_excel(self, json_data, output_file):
        """
        Converts JSON data to an Excel file.

        Args:
            json_data (dict): The JSON data to be converted.
            output_file (str): The path to the output Excel file.
        """
        # 创建Excel写入器
        writer = pd.ExcelWriter(output_file, engine='openpyxl')

        # 遍历顶层的数据集（如ASDIV, code1等）
        for dataset_name, dataset in json_data.items():
            # 循环处理top5和top1的数据
            for top_key in ['top5', 'top1']:
                # 创建一个DataFrame存储当前top_key的数据
                data_frames = {}

                # 遍历每一层数据
                for layer, metrics in dataset.items():
                    # 将当前层的数据转换为DataFrame
                    df = pd.DataFrame(metrics[top_key], index=[layer])
                    if layer in data_frames:
                        data_frames[layer] = pd.concat([data_frames[layer], df])
                    else:
                        data_frames[layer] = df

                # 合并所有层的DataFrame
                final_df = pd.concat(data_frames.values())

                # 生成工作表名称，并确保不超过31个字符
                sheet_name = f'{dataset_name}_{top_key}'
                if len(sheet_name) > 31:
                    # 如果名称超过31个字符，需要截断，因为Excel的工作表名称最多只能有31个字符
                    sheet_name = f"{dataset_name[:19]}_{top_key}"
                print(sheet_name)

                # 写入一个新的Sheet，确保名称长度不超过31个字符
                final_df.to_excel(writer, sheet_name=sheet_name)

        # 关闭Excel写入器，保存文件
        writer.close()

    def generate_bar_chart_from_excel(self,
                                      input_file,
                                      sheet_name_key_words: list[str] = None,
                                      save_pic_dir: str = None,
                                      limit_y: int = None):
        """
        Generates bar charts from an Excel file, with specific colors for different codes.

        Args:
            input_file (str): The path to the input Excel file.
            sheet_name_key_words (list[str], optional): A list of keywords to filter sheet names. Defaults to None.
            save_pic_dir (str, optional): The directory where the bar charts will be saved. Defaults to None.
            limit_y (int, optional): The maximum value for the y-axis. Defaults to None.
        """
        # 读取Excel文件中的所有工作表
        xls = pd.ExcelFile(input_file)

        # 定义颜色映射
        color_map = {'code_high': 'orange', 'code_medium': 'blue', 'code_low': 'green'}

        if save_pic_dir:
            os.makedirs(save_pic_dir, exist_ok=True)

        # 遍历每个工作表
        for sheet_name in xls.sheet_names:
            # 如果指定了关键字，只处理包含关键字的工作表，表名一般为：LeeTCode_code_high_top1 这样
            if sheet_name_key_words is not None and all(
                    sheet_key_word not in sheet_name for sheet_key_word in sheet_name_key_words):
                continue

            # 读取当前工作表的数据，将第一列设置为索引
            df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)

            # 对列进行排序，以便 code_high, code_medium, code_low 的顺序
            df = df[['code_high', 'code_medium', 'code_low']]

            # 获取列名并映射颜色
            colors = [color_map.get(col, 'gray') for col in df.columns]

            # 绘制柱状图
            plt.figure(figsize=(10, 6))  # 增加图表尺寸
            ax = df.plot(kind='bar', width=0.8, color=colors)
            ax.set_title(sheet_name)  # 设置图表标题为工作表名称
            ax.set_xlabel('Layer')  # 设置x轴标签
            ax.set_ylabel('Values')  # 设置y轴标签
            if limit_y is not None:
                ax.set_ylim(0, limit_y)  # 设置y轴范围为0到100

            # 设置x轴标签，使用90度旋转
            ax.set_xticklabels(df.index, rotation=90, ha='center')

            # 显示图例，调整布局以防止标签被截断
            plt.legend(title='Codes')
            plt.tight_layout()

            # 保存图表
            if save_pic_dir is not None:
                plt.savefig(os.path.join(save_pic_dir, f'{sheet_name}.png'))
                plt.close()

            plt.show()

    def analyse_layer(self,
                      excel_file,
                      save_folder,
                      top1_or_top5: str = "top1"):
        """
        Analyzes the layer data.
        仅计算top1的数据。code_high, code_medium, code_low 分开计算，综合考量每一层跨越不同数据集得到的结果，计算占比的平均值和方差。

        Args:
            excel_file (str):


        """
        xls = pd.ExcelFile(excel_file)

        # 字典，读取了所有的数据，第一层key是训练集数据的名字，第二层key是layer，value为列表
        train_dataset_names = ["code_high", "code_medium", "code_low"]
        all_data = {"code_high": {}, "code_medium": {}, "code_low": {}}

        # 把所有的数据加载到all_data中
        for ii, sheet_name in enumerate(xls.sheet_names):
            # 仅处理top1 或 top5的 sheet_name
            if top1_or_top5 not in sheet_name:
                continue
            print("in analyse_layer(), sheet_name: ", sheet_name)

            # 读取当前工作表的数据，将第一列设置为索引
            df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0, header=0)

            # 获取数据
            layer_names = df.index.to_list()
            for layer_name in layer_names:
                for train_set_name in train_dataset_names:
                    # 如果没有这个key（layer的名字，excel的行名），就创建一个空列表
                    if layer_name not in all_data[train_set_name]:
                        all_data[train_set_name][layer_name] = []

                    # 处理nan的特殊情况
                    loc_val = df.loc[layer_name, train_set_name]
                    loc_val = loc_val if (not np.isnan(loc_val)) else 0
                    all_data[train_set_name][layer_name].append(loc_val)

        # 把数据处理成平均值和方差
        processed_mean = {"code_high": {}, "code_medium": {}, "code_low": {}}
        processed_varience = {"code_high": {}, "code_medium": {}, "code_low": {}}
        processed_std = {"code_high": {}, "code_medium": {}, "code_low": {}}
        for train_set_name in all_data.keys():
            for layer_name in all_data[train_set_name].keys():
                processed_mean[train_set_name][layer_name] = statistics.mean(all_data[train_set_name][layer_name])
                processed_varience[train_set_name][layer_name] = statistics.variance(all_data[train_set_name][layer_name])

                # 两种计算标准差的方法，一个是样本标准差stdev，一个是总体标准差pstdev
                processed_std[train_set_name][layer_name] = statistics.pstdev(all_data[train_set_name][layer_name])
                # processed_std[train_set_name][layer_name] = statistics.stdev(all_data[train_set_name][layer_name])

        # 将结果写入Excel文件
        mean_file = os.path.join(save_folder, 'layer_mean.xlsx')
        var_file = os.path.join(save_folder, 'layer_variance.xlsx')
        sed_file = os.path.join(save_folder, 'layer_std.xlsx')

        pd.DataFrame(processed_mean).to_excel(mean_file, index=True)
        pd.DataFrame(processed_varience).to_excel(var_file, index=True)
        pd.DataFrame(processed_std).to_excel(sed_file, index=True)


if __name__ == "__main__":
    need_write_to_excel = False
    need_generate_bar_chart = False
    need_analyse_layer = True

    dataset_names = ["LeeTCode_code_high", "LeeTCode_code_medium", "LeeTCode_code_low",
                     "olympic_OE_TO_maths_en_COMP", "olympic_OE_TO_physics_en_COMP",
                     "olympic_TP_TO_maths_en_COMP", "olympic_TP_TO_physics_en_COMP",
                     "GSM", "MultiArith", "SVAMP", "ASDiv"]

    layer_analyser = LayerAnalyser(
        dataset_name_list=dataset_names,
        model_name="llama2-7b",
        sample_size=5,
        base_data_dir=r"D:\Software_data\Pycharm_prj\LLM_Gradient\proj_less\score",
        save_dir=r"D:\Software_data\Pycharm_prj\LLM_Gradient\proj_gradient_analysis\analyse_result")

    # 将数据写入Excel文件
    if need_write_to_excel:
        layer_analyser.trans_to_excel(layer_analyser.dataset_data,
                                      os.path.join(layer_analyser.save_dir, "dataset_count_ori.xlsx"))
        layer_analyser.trans_to_excel(layer_analyser.percent_dataset_data,
                                      os.path.join(layer_analyser.save_dir, "dataset_count_percent.xlsx"))

    # 根据得到的excel文件生成柱状图，此处是生成top1和top5的柱状图，用的是100比的数据
    if need_generate_bar_chart:
        layer_analyser.generate_bar_chart_from_excel(
            input_file=os.path.join(layer_analyser.save_dir, "dataset_count_percent.xlsx"),
            sheet_name_key_words=["top1", "top5"],
            save_pic_dir=os.path.join(layer_analyser.save_dir, "bar_chart"),
            limit_y=100)

    if need_analyse_layer:
        layer_analyser.analyse_layer(
            os.path.join(layer_analyser.save_dir, "dataset_count_percent.xlsx"),
            save_folder=layer_analyser.save_dir,
            top1_or_top5="top1")

    print("process ended")
