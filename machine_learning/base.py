"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
import numpy as np
from .validate import check_is_trained, check_predict_data, check_train_data
import six
import abc


class BaseClassifyModel(six.with_metaclass(abc.ABCMeta)):
    """Base class of classification model, classification model
    Have train and predict model.
    """

    def train(self, X, y):
        """Train model"""
        X, y = check_train_data(X, y)

        if hasattr(self, "_train"):
            return self._train(X, y)
        else:
            raise Exception(
                "The subclass derived from base model not implement "
                "`_train` method."
            )

    def predict(self, X):
        """Predict samples"""

        check_is_trained(self, ["n_samples", "n_features"])
        X = check_predict_data(self, X)

        if hasattr(self, "_predict"):
            return self._predict(X)
        else:
            raise Exception(
                "The subclass derived from base model not implement "
                "`_predict` method."
            )


class LinearModel(six.with_metaclass(abc.ABCMeta)):
    """Base class for Linear Models

    formula,
        original linear model: y = w0 + w1*x1 + ... + wp*xp
    """

    def check_transform_X_y(self, X, y):
        """return X, y"""
        return X, y

    def train(self, X, y):
        """Train Linear model."""

        X, y = check_train_data(X, y)

        if hasattr(self, "_train"):
            return self._train(X, y)
        else:
            raise Exception(
                "The subclass derived from base model not implement "
                "`_train` method."
            )

    def predict(self, X):
        """Predict samples"""

        check_is_trained(self, "_coef")
        X = check_predict_data(self, X)

        if hasattr(self, "_predict"):
            return self._predict(X)
        else:
            raise Exception(
                "The subclass derived from base model not implement "
                "`_predict` method."
            )

class ClusterModel(six.with_metaclass(abc.ABCMeta)):
    """Base cluster class."""

    def train(self, X):
        """Cluster unlabeled data set."""

        if isinstance(X, list):
            X = np.array(X)
        assert isinstance(X, np.ndarray)
        if X.ndim != 2:
            raise TypeError("Train data X should be a 2-d arrays.")

        if hasattr(self, "_train"):
            return self._train(X)
        else:
            raise Exception(
                "The subclass derived from base model not implement "
                "`_train` method."
            )

    def predict(self, X):
        """Predict samples."""

        X = check_predict_data(self, X)

        if hasattr(self, "_predict"):
            return self._predict(X)
        else:
            raise Exception(
                "The subclass derived from base model not implement "
                "`_predict` method."
            )
