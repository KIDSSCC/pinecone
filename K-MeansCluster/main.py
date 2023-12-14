import matplotlib.pyplot as plt
import KMeans as kmeans
from sklearn.cluster import KMeans
from sklearn import datasets


X = None


def print_something(x):
    print('type is: ', type(x))
    print('value is: ', x)


# 模型定义
def model(n_clusters):
    estimator = KMeans(n_clusters)
    return estimator


# 模型训练
def train(estimator):
    estimator.fit(X)


def kmeans_sklearn():
    iris = datasets.load_iris()
    global X
    X = iris.data[:, :4]
    '''
    # 取前两个维度，绘制数据分布图
    plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=2)
    plt.show()
    '''
    # 初始化实例，训练拟合
    estimator = model(3)
    train(estimator)
    # 获取聚类标签
    label_pred = estimator.labels_
    # 绘制聚类结果
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label_1')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label_2')
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label_3')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=2)
    plt.show()


def kmeans_handwritten():
    iris = datasets.load_iris()
    global X
    X = iris.data[:, :4]
    centroids, clusterAssment = kmeans.kmeans(X, 3)
    kmeans.draw(X, centroids, clusterAssment)


if __name__ == '__main__':
    print('KMeans聚类算法实现鸢尾花聚类')
    kmeans_handwritten()

