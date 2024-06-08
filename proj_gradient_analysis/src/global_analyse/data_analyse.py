import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt

def read_pt_and_trans_to_array(path):
    # 读入torch的pt文件，转换为numpy数组
    torch_tensor = torch.load(path)
    torch_tensor = torch_tensor.to(torch.float32)
    numpy_array = torch_tensor.numpy()

    return numpy_array

data1 = read_pt_and_trans_to_array(r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\code_high-ckpt7-adam\dim8192\all_orig.pt")
data2 = read_pt_and_trans_to_array(r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\code_low-ckpt7-adam\dim8192\all_orig.pt")
data3 = read_pt_and_trans_to_array(r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\code_medium-ckpt7-adam\dim8192\all_orig.pt")



def descriptive_stats(data, name):
    mean_vector = np.mean(data, axis=0)
    std_dev_vector = np.std(data, axis=0)
    min_vector = np.min(data, axis=0)
    max_vector = np.max(data, axis=0)

    print(f"Descriptive Statistics for {name}:")
    print(f"Mean of means: {np.mean(mean_vector)}")
    print(f"Mean of standard deviations: {np.mean(std_dev_vector)}")
    print(f"Overall min: {np.min(min_vector)}")
    print(f"Overall max: {np.max(max_vector)}")
    print()

    return mean_vector, std_dev_vector, min_vector, max_vector

# 获取每个数据集的统计描述
mean_vector1, std_dev_vector1, min_vector1, max_vector1 = descriptive_stats(data1, "Dataset 1")
mean_vector2, std_dev_vector2, min_vector2, max_vector2 = descriptive_stats(data2, "Dataset 2")
mean_vector3, std_dev_vector3, min_vector3, max_vector3 = descriptive_stats(data3, "Dataset 3")

# 计算并输出向量之间的欧氏距离
def compare_vectors(v1, v2, name1, name2, description):
    distance = np.linalg.norm(v1 - v2)
    print(f"Euclidean distance between {description} of {name1} and {name2}: {distance}")


mo1 = np.linalg.norm(mean_vector1)
mo2 = np.linalg.norm(mean_vector2)
mo3 = np.linalg.norm(mean_vector3)

print(f"Mean vector 1 norm: {mo1}")
print(f"Mean vector 2 norm: {mo2}")
print(f"Mean vector 3 norm: {mo3}")

compare_vectors(mean_vector1, mean_vector2, "Dataset 1", "Dataset 2", "mean vectors")
compare_vectors(mean_vector1, mean_vector3, "Dataset 1", "Dataset 3", "mean vectors")
compare_vectors(mean_vector2, mean_vector3, "Dataset 2", "Dataset 3", "mean vectors")

compare_vectors(std_dev_vector1, std_dev_vector2, "Dataset 1", "Dataset 2", "standard deviation vectors")
compare_vectors(std_dev_vector1, std_dev_vector3, "Dataset 1", "Dataset 3", "standard deviation vectors")
compare_vectors(std_dev_vector2, std_dev_vector3, "Dataset 2", "Dataset 3", "standard deviation vectors")