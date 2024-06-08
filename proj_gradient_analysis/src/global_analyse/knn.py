import torch
import numpy as np


def read_pt_and_trans_to_array():
    # 读入torch的pt文件，转换为numpy数组
    torch_tensor = torch.load('your_file.pt')
    numpy_array = torch_tensor.numpy()

    # 检查大小是否一致
    print("Torch tensor size:", torch_tensor.size())
    print("Numpy array size:", numpy_array.shape)

    return numpy_array



import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 假设vecs是大小为[1700, 8192]的向量集
vecs = np.random.rand(1700, 8192)

# 假设data是大小为[x, 8192]的数据集
data = np.random.rand(500, 8192)  # 假设x=500

# 创建k近邻分类器
knn = KNeighborsClassifier(n_neighbors=1)

# 训练分类器
knn.fit(vecs, np.arange(len(vecs)))

# 预测数据集所属的向量集
predicted_labels = knn.predict(data)

# 统计预测结果
votes = np.bincount(predicted_labels)
most_similar_vector_set_index = np.argmax(votes)

print("Data set belongs to vector set:", most_similar_vector_set_index)
