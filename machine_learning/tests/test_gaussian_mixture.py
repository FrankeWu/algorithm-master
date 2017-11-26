"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from machine_learning import cluster


X1, Y1 = make_blobs(n_features=2, centers=3)
model = cluster.GaussianMixtureModel(max_iter=100, n_clusters=3).train(X1)
predict_label = model.predict(X1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=predict_label, s=25, edgecolor='k')
plt.show()
