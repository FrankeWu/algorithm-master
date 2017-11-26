"""
Author ： g faia
Email : gutianfeigtf@163.com
------------------------------
"""
import numpy as np
from .base import LinearModel


def least_squares_estimator(X, y):
    """Transform X to a list, and, y should also be a list.
    return, (bias, coef)
    """
    if X.ndim != 1: X = [x[0] for x in X]
    if type(y) == list: y = list(y)
    if len(X) == len(y):
        n_samples = len(X)
    else:
        raise Exception("The length of X and y should be the same.")

    ave_X = sum(X) / n_samples
    ave_y = sum(y) / n_samples
    sum_X_y = sum([X[i] * y[i] for i in range(n_samples)])
    sum_X_2 = sum([X[i] ** 2 for i in range(n_samples)])
    coef = (sum_X_y - ave_y * sum(X))/ (sum_X_2 - ave_X * sum(X))
    bias = (ave_y * sum_X_2 - ave_X * sum_X_y) / (sum_X_2 - ave_X * sum(X))

    return bias, coef


class LinearRegression(LinearModel):
    """Linear regression algorithm, use least square method
    (lsm) for object function.

    object function, min || X * w - Y ||2
    formula, y = w * x , w = [bias, w1, w2, ... , wn]
    """
    def __init__(self, learn_rate=0.01, init_coef=.0, init_bias=.0,
                 max_iter=500):

        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.bias = init_bias
        self.coef = init_coef

    def init_coefficient(self):
        """_coef is transformed.

        _coef, [bias, w1, w2, ..., wn]
        """
        return [self.bias] + [self.coef for i in range(self.n_features)]

    def check_transform_X_y(self, X, y):
        """Transform X, y"""
        X = np.array([[1] + list(X[i]) for i in range(self.n_samples)])
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        return X, y

    def _train(self, X, y):
        """Minimize the cost function by using gradient
        descent algorithm."""
        self.n_samples, self.n_features = n_samples, n_features = X.shape
        _coef = np.array(self.init_coefficient())
        max_iter = self.max_iter
        learn_rate = self.learn_rate
        # X, y = self.check_transform_X_y(X, y)

        # When solve linear model y = b + a*x
        # Use Least Squares Estimator to optimize model
        if n_features == 1:
            _coef = list(least_squares_estimator(X, y))
        else:
            X, y = self.check_transform_X_y(X, y)

            for i in range(max_iter):
                coef = []
                for j in range(n_features + 1):
                    descent = 0
                    for k in range(n_samples):
                        descent += 2 * (np.sum(np.multiply(X[k], _coef)) - y[k]) * X[k][j]
                    coef.append(
                        _coef[j] - 1 / n_samples * learn_rate * descent
                    )
                _coef = coef

        self._coef = _coef
        return self

    def _predict(self, X):
        """Use model to estimate the unknown samples."""

        _coef = self._coef
        y_ = []
        samples = X.shape[0]
        X = np.array([[1] + list(X[i]) for i in range(samples)])

        for i in range(samples):
            y_.append(np.sum(np.multiply(X[i], _coef)))

        return y_
