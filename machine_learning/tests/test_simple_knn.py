"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import generate_data
from machine_learning import classification

x1, y1 = generate_data.generate_random_dataset()

model = classification.SimpleKNNModel().train(x1, y1)

point = np.array([0, 0])
y = model.predict(point)

plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)
plt.scatter(point[0], point[1], marker='*', c=y)
plt.show()
