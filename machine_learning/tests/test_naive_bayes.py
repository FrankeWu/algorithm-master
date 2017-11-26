"""
Author ： g faia
Email : gutianfeigtf@163.com

From `statistics study method` p 52
"""
import numpy as np
from machine_learning import classification


x = np.array([[1, 1], [1, 2], [1, 2], [1, 1], [1, 1], [2, 1], [2, 2], [2, 2],
              [2, 3], [2, 3], [3, 3], [3, 2], [3, 2], [3, 3], [3, 3]])
y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
model = classification.NaiveBayesBModel().train(x, y)
label = model.predict(np.array([[2, 1], [2, 5]]))
print(label)
