import matplotlib.pyplot as plt
import numpy as np


def cal_distance(x, y):
    """
    计算两个点之间的欧式距离
    :param x:元素x
    :param y:元素y
    :return:两个元素之间的欧式距离
    """
    return np.sqrt(np.sum(x-y)**2)


def rand_cent(dataset, k):
    """
    根据给定的数据集，从中选取k个点作为初始聚簇质心
    :param dataset:包含所有点的数据集
    :param k:聚类数目
    :return:初始聚簇质心
    """
    m, n = dataset.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))
        centroids[i, :] = dataset[index, :]
    return centroids


def kmeans(dataset, k):
    """
    对数据集dataset进行k聚类
    :param dataset:包含所有点的数据集
    :param k:进行k聚类
    :return:质心的坐标，每个点所属的聚类
    """
    m = np.shape(dataset)[0]
    # 创建一个辅助数组，对应数据集中的每个点，包含两列，第一列为该点所属的聚簇，第二列为该点到质心的距离
    cluster_assment = np.mat(np.zeros((m, 2)))
    cluster_change = True
    # 1. 初始化质心
    centroids = rand_cent(dataset, k)
    while cluster_change:
        # 当两次迭代之间，每个点所属的聚簇不发生变化时，停止迭代
        cluster_change = False
        # 2. 计算每个点所属的质心
        for i in range(m):
            min_dist = 100000.0
            min_index = -1
            # 对于每个点，计算其与所有质心之间的距离，选取距离最小的，加入到对应的聚类中
            for j in range(k):
                distance = cal_distance(centroids[j, :], dataset[i, :])
                if distance<min_dist:
                    min_dist = distance
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_change = True
                cluster_assment[i, :] = min_index, min_dist**2
        # 3. 更新质心
        for j in range(k):
            '''
            cluster_assment[:, 0]创建了一个切片，包含了cluster_assment所有行的第一列，通过.A将其转换为了一个array
            cluster_assment[:, 0].A == j，将上一行提到的array与j进行比较。得到的结果仍然是一个array，每一个位置为对位比较的结果，0或1
            通过np.nonzero，获取非0元素的下标，即所属聚簇为j的点的下标
            '''
            points_in_cluster = dataset[np.nonzero(cluster_assment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(points_in_cluster, axis=0)
    print('cluster complete')
    return centroids, cluster_assment


def draw(data, center, assment):
    """
    进行绘图演示
    :param data:原始点的信息
    :param center:聚簇中心点的信息
    :param assment:每个点所属的聚簇信息
    :return:
    """
    length = len(center)
    fig = plt.figure
    data1 = data[np.nonzero(assment[:, 0].A == 0)[0]]
    data2 = data[np.nonzero(assment[:, 0].A == 1)[0]]
    data3 = data[np.nonzero(assment[:, 0].A == 2)[0]]
    # 选取前两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 0], data1[:, 1], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 0], data2[:, 1], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 0], data3[:, 1], c="blue", marker='+', label='label2')
    # 绘制质心
    for i in range(length):
        plt.annotate('center', xy=(center[i, 0], center[i, 1]), xytext=(center[i, 0] + 1, center[i, 1] + 1), arrowprops=dict(facecolor='yellow'))
    plt.show()

    '''
    # 选取后两个维度绘制原始数据的散点图
    plt.scatter(data1[:,2],data1[:,3],c="red",marker='o',label='label0')
    plt.scatter(data2[:,2],data2[:,3],c="green", marker='*', label='label1')
    plt.scatter(data3[:,2],data3[:,3],c="blue", marker='+', label='label2')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center',xy=(center[i,2],center[i,3]),xytext=(center[i,2]+1,center[i,3]+1),arrowprops=dict(facecolor='yellow'))
    plt.show()
    '''