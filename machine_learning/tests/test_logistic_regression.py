"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
import numpy as np
from machine_learning.classification import BinLogisticRegressionModel
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


x1, y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, n_samples=100)
x_train = x1[0:70]
y_train = y1[0:70]
x_test = x1[70:]
y_test = y1[70:]
model = BinLogisticRegressionModel().train(x_train, y_train)

print(model._coef)

weights = model._coef
x = np.arange(-10, 10)
y = []
for _ in x:
    y.append((- (weights[0] / weights[1]) - (weights[2] / weights[1]) * _))


# draw the classification linear.
plt.figure()
plt.plot(x, y)
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)
plt.show()


def test_accuracy(a, b):
    """The accuracy of predict, the similarity of two lists.
    """
    if len(a) != len(b):
        raise Exception("Two lists should have the same length.")

    length = len(a)
    correct = 0

    for i in range(length):
        if a[i] == b[i]:
            correct += 1

    return correct / length

predict = model.predict(x_test)
print(test_accuracy(predict, y_test))
