"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
from sklearn.datasets import make_classification
from machine_learning import classification
import matplotlib.pyplot as plt
import numpy as np

# generate fake data set
plt.figure()
x1, y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, n_samples=50)
x_train, y_train = x1[:40], y1[:40]
x_test, y_test = x1[40:], y1[40:]

# train perceptron model
perceptron_model = classification.PerceptronModel(max_iter=1000, learn_rate=0.01)\
    .train(x_train, y_train)

weights = perceptron_model._coef
x = np.arange(-5, 5)
y = []
for _ in x:
    y.append((-(weights[2] / weights[1]) * _ - (1 / weights[1]) * weights[0]))
plt.plot(x, y)

print(y1)
# plot data set
plt.scatter(x_test[:, 0], x_test[:, 1], marker='o', c=y_test)
plt.show()
