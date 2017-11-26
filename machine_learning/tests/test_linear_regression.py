"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
from machine_learning import regression
import numpy as np

# test simple function y = x
X = np.array([[1], [2], [3], [4], [6], [8], [10]])
y = np.array([0.9, 2.1, 3, 4, 6.2, 8, 10])
model = regression.LinearRegression().train(X, y)
print(model._coef)
