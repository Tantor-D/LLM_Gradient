import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch


def read_pt_and_trans_to_array(path):
    # 读入torch的pt文件，转换为numpy数组
    torch_tensor = torch.load(path)
    torch_tensor = torch_tensor.to(torch.float32)
    numpy_array = torch_tensor.numpy()

    # 检查大小是否一致
    print("Torch tensor size:", torch_tensor.size())
    print("Numpy array size:", numpy_array.shape)

    return numpy_array

# 假设 data1, data2, data3 是你的三个已经降维到8192维的数据集
# 随机生成数据作为示例
np.random.seed(0)


data1 = read_pt_and_trans_to_array(r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\code_high-ckpt7-adam\dim8192\all_orig.pt")
data2 = read_pt_and_trans_to_array(r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\code_low-ckpt7-adam\dim8192\all_orig.pt")
data3 = read_pt_and_trans_to_array(r"E:\backup_for_servicer\1_project\project_1700_code\grads\llama2-7b-p0.05-lora-seed3\code_medium-ckpt7-adam\dim8192\all_orig.pt")


# 合并数据集
data = np.vstack((data1, data2, data3))
# 初始化标签，便于比较聚类前后的变化
initial_labels = np.array([0]*1700 + [1]*1700 + [2]*1700)
# 应用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
# 聚类结果标签
labels = kmeans.labels_
# 显示聚类前后的标签对比
print("Initial Labels (expected clusters):")
print(initial_labels)
print("Labels After K-Means Clustering:")
print(labels)
# 可视化聚类结果，只能用降维后的数据进行可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=initial_labels, cmap='viridis', alpha=0.5)
plt.title('Before Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.subplot(1, 2, 2)
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('After K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.show()


