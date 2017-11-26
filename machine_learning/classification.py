"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
import numpy as np
from .distance import get_distance_metric
from .base import BaseClassifyModel, LinearModel
from collections import Counter


class SimpleKNNModel(BaseClassifyModel):
    """KNN algorithm use distance between samples to define
    similarity. Utilize the most proportion labels in the
    nearest k's samples to label the unknown sample.
    """
    def __init__(self, dist="euclidean", k_value=10):

        self.distance_metric = get_distance_metric(dist)
        self.k = k_value

    def _train(self, X, y):
        """The simple KNN algorithm nearly no need train operation.
        The information is contained in data set.

        X, 2-d numpy.arrays
        y, 1-d numpy.arrays
        """
        self.n_samples, self.n_features = X.shape
        if self.n_samples == 0:
            raise Exception("The number of samples cannot be zero.")

        self.X, self.y = X, y
        return self

    def _predict(self, X):
        """
        X, 2-d numpy.array

        return unpredicted samples' labels.
        """
        n_samples = self.n_samples
        X_model, y_model = self.X, self.y
        samples = X.shape[0]
        k_value = self.k
        distance_metric = self.distance_metric
        y_ = []

        for i in range(samples):
            nearest_k = [distance_metric(X_model[j], X[i])
                         for j in range(n_samples)]

            arg_index = np.argsort(nearest_k)[:k_value]
            labels = [y_model[index] for index in arg_index]
            max_labels = -1
            counter = Counter(labels)

            for key in counter.keys():
                if counter.get(key) > max_labels:
                    max_labels = key

            y_.append(max_labels)

        return y_


class NaiveBayesBModel(BaseClassifyModel):
    """naive Bayes classifiers are a family of simple probabilistic
    classifiers based on applying Bayes' theorem with strong (naive)
    independence assumptions between the features.

    The multinomial naive bayes version can only train discrete features.

    formula: use Laplace smoothing to estimate probability.
    """

    def __init__(self, lambda_value=1):

        self.lambda_value = lambda_value

    def _train(self, X, y):
        """naive Bayes learning probability from data set.

        X, 2-d numpy.arrays
        y, 1-d numpy.arrays
        """
        self.n_samples, self.n_features = n_samples, n_features = X.shape
        lambda_value = self.lambda_value

        # calculate prior probability and condition probability
        self.labels_counter = labels_counter = Counter(y)
        self.labels_set = labels_set = labels_counter.keys()
        prior_probability = {}

        for l in labels_set:
            prior_probability[l] = \
                (labels_counter.get(l) + lambda_value) \
                / (n_samples + len(labels_counter) * lambda_value)

        x_split = {l: [] for l in labels_set}

        for i in range(n_samples):
            x_split[y[i]].append(list(X[i]))

        for l in x_split.keys():
            x_split[l] = np.array(x_split[l])

        condition_probability = {l: {} for l in labels_set}
        x_total_set = []

        for i in range(n_features):
            x_total_set.append(len(set(X[:, i])))

        for l in labels_set:
            for i in range(n_features):
                x_set = condition_probability[l][i + 1] = {}
                x_counter = Counter(x_split[l][:, i])
                for x in x_counter.keys():
                    x_set[x] = (x_counter[x] + lambda_value) \
                               / (labels_counter[l] + x_total_set[i] * lambda_value)

        self.prior_probability = prior_probability
        self.condition_probability = condition_probability
        self.x_total_set = x_total_set
        return self

    def _predict(self, X):
        """Use probability to predict unlabeled samples.

        X, 2-d numpy.array

        return unpredicted samples' labels.
        """
        n_features = self.n_features
        prior_probability = self.prior_probability
        condition_probability = self.condition_probability
        samples = X.shape[0]
        y_ = []
        labels_set = self.labels_set
        labels_counter = self.labels_counter
        x_total_set = self.x_total_set
        lambda_value = self.lambda_value

        for i in range(samples):
            max_p = 0
            max_label = None
            sample = X[i]
            for l in labels_set:
                p = prior_probability[l]
                for j in range(n_features):
                    try:
                        sub = condition_probability[l][j + 1][sample[j]]
                    except KeyError:
                        sub = lambda_value / (labels_counter[l] + x_total_set[j] * lambda_value)
                    p = p * sub
                if p > max_p:
                    max_p = p
                    max_label = l
            y_.append(max_label)

        return y_

class PerceptronModel(LinearModel):
    """Perceptron is a simple linear model to classify samples.
    PerceptronModel is the original version.

    formula,
        y = sign(w * x + b) = sign(w0 + w1*x1 + ... + wn*xn)
        if sign(z) > 0: y = +1
        else: y = -1
    """

    def __init__(self, max_iter=1000, init_coef=.0, init_bias=.0,
                 learn_rate=0.5):

        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.coef = init_coef
        self.bias = init_bias

    def init_coefficient(self):
        """_coef is transformed.

        _coef, [bias, w1, w2, ..., wn]
        """
        return [self.bias] + [self.coef for i in range(self.n_features)]

    def check_transform_X_y(self, X, y):
        """Check and transform (X, y) for perceptron algorithm."""

        # The value of y should be -1 or +1
        # Always, the set of y will be {0, 1}, transform 0 to -1
        if set(y) == {0, 1}:
            for i in range(self.n_samples):
                if y[i] == 0: y[i] = -1

        # transform x to [1, x]
        X = np.array([[1] + list(X[i]) for i in range(self.n_samples)])
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        return X, y

    def _train(self, X, y):
        """Train the weights w and bias b, parameters of model.

        X, 2-d numpy.arrays
        y, 1-d numpy.arrays and the range of y is {+1, -1}
        """

        self.n_samples, self.n_features = n_samples, n_features = X.shape
        _coef = np.array(self.init_coefficient())
        max_iter = self.max_iter
        learn_rate = self.learn_rate
        X, y = self.check_transform_X_y(X, y)

        for i in range(max_iter):
            for j in range(n_samples):
                if np.dot(X[j], _coef) <= 0:
                    _coef += np.float64(learn_rate) * y[j] * X[j]

        self._coef = _coef
        return self

    def _predict(self, X):
        """
        X, 2-d numpy.array

        return unpredicted samples' labels.
        """
        _coef = self._coef
        X = np.array([[1] + list(X[i]) for i in range(self.n_samples)])
        X = X.astype(np.float64)
        return [1 if s > 0 else -1 for s in np.dot(X, _coef)]


class BinLogisticRegressionModel(LinearModel):
    """Logistic regression is not a regression model,
    It is a classification model use logit function.

    formula，
        p(y = 1 | x) = logistic(w * x)
        w = [bias, w1, w2, ..., wn], n is n_features
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
        """Check and transform (X, y) for Logistic algorithm."""
        if set(y) != {0, 1}:
            raise Exception(
                "For binary logistic regression model, "
                "set of y should be {0, 1}"
            )

        X = np.array([[1] + list(X[i]) for i in range(self.n_samples)])
        return X, y

    @staticmethod
    def logistic_func(x, coef):
        """standard logistic function
        formula: exp(w*x)/(1+exp(w*x)
        """
        sub = np.exp(- np.sum(np.multiply(x, coef)))
        return 1 / (1 + sub)

    def _train(self, X, y):
        """Minimize the cost function by using gradient
        descent algorithm.

        X, 2-d numpy.arrays
        y, 1-d numpy.arrays
        """
        self.n_samples, self.n_features = n_samples, n_features = X.shape
        max_iter = self.max_iter
        learn_rate = self.learn_rate
        X, y = self.check_transform_X_y(X, y)
        _coef = self.init_coefficient()

        for i in range(max_iter):
            coef = []
            for j in range(n_features + 1):
                descent = 0
                for k in range(n_samples):
                    descent += (self.logistic_func(X[k], _coef) - y[k]) * X[k][j]
                coef.append(
                    _coef[j] - 1 / n_samples * learn_rate * descent
                )
            _coef = coef

        self._coef = _coef
        return self

    def _predict(self, X):
        """Use logistic linear model to predict labels."""

        _coef = self._coef
        y_ = []
        samples = X.shape[0]
        X = np.array([[1] + list(X[i]) for i in range(samples)])

        for i in range(samples):
            positive = self.logistic_func(X[i], _coef)
            negative = 1 - positive
            if positive >= negative: y_.append(1)
            else: y_.append(0)

        return y_

