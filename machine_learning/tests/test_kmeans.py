"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from machine_learning import cluster


X1, Y1 = make_blobs(n_features=2, centers=3)
model = cluster.KMeansCluster(max_iter=100).train(X1)
centers = model.centers
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], marker='*')
plt.show()
