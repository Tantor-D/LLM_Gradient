import os
import json
import copy
import pandas as pd
import matplotlib.pyplot as plt


class LayerAnalyser:
    """
    A class used to analyze layer data from JSON files and visualize it in Excel and bar charts.

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

    def generate_bar_chart_from_excel(self, input_file, sheet_name_key_words: list[str] = None):
        """
        Generates bar charts from an Excel file, with specific colors for different codes.

        Args:
            input_file (str): The path to the input Excel file.
            sheet_name_key_words (list[str], optional): A list of keywords to filter sheet names. Defaults to None.
        """
        # 读取Excel文件中的所有工作表
        xls = pd.ExcelFile(input_file)

        # 定义颜色映射
        color_map = {'code_high': 'orange', 'code_medium': 'blue', 'code_low': 'green'}

        # 遍历每个工作表
        for sheet_name in xls.sheet_names:
            # 如果指定了关键字，只处理包含关键字的工作表
            flag_ignore = False
            if sheet_name_key_words is not None:
                flag_ignore = True
                for sheet_key_word in sheet_name_key_words:
                    if sheet_key_word in sheet_name:
                        flag_ignore = False
            if flag_ignore:
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

            # 设置x轴标签，使用90度旋转
            ax.set_xticklabels(df.index, rotation=90, ha='center')

            # 显示图例，调整布局以防止标签被截断
            plt.legend(title='Codes')
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    need_write_to_excel = True

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

    # 根据得到的excel文件生成柱状图
    layer_analyser.generate_bar_chart_from_excel(os.path.join(layer_analyser.save_dir, "dataset_count_percent.xlsx"),
                                                 ["top5"])

    print("aaa")
