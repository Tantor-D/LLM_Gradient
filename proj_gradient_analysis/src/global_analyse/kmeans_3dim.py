import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
def read_pt_and_trans_to_array(path):
    # 读入torch的pt文件，转换为numpy数组
    torch_tensor = torch.load(path)
    torch_tensor = torch_tensor.to(torch.float32)
    numpy_array = torch_tensor.numpy()

    return numpy_array
# 假设 data1, data2, data3 是你的三个数据集
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

# 使用PCA进行降维到3维
pca = PCA(n_components=3)
data_reduced = pca.fit_transform(data)

# 创建3D散点图来可视化降维后的数据
fig = plt.figure(figsize=(14, 7))

# 前聚类3D可视化
ax1 = fig.add_subplot(121, projection='3d')
scatter = ax1.scatter(data_reduced[:, 0], data_reduced[:, 1], data_reduced[:, 2], c=initial_labels, cmap='viridis', alpha=0.5)
ax1.set_title('Before Clustering')
ax1.set_xlabel('PCA 1')
ax1.set_ylabel('PCA 2')
ax1.set_zlabel('PCA 3')

# 聚类后3D可视化
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(data_reduced[:, 0], data_reduced[:, 1], data_reduced[:, 2], c=labels, cmap='viridis', alpha=0.5)
ax2.set_title('After K-Means Clustering')
ax2.set_xlabel('PCA 1')
ax2.set_ylabel('PCA 2')
ax2.set_zlabel('PCA 3')

# 显示图形
plt.show()