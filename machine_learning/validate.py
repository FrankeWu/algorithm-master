"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
import numpy as np


def check_is_trained(model, attributes):
    """Check whether model is trained. Some attributes are not
    initialized before trained.
    """
    if not hasattr(model, "train"):
        raise TypeError("%s is not an model instance." % model)

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all([hasattr(model, attr) for attr in attributes]):
        raise Exception(
            "This %s's instance is not trained yet. Call `train` "
            "method before predict." % type(model).__name__
        )


def check_train_data(X, y):
    """Check type of data used to train model. This method
    accept `list` type.

    n_samples, n_features, X.shape
        X's samples should fit to y's.
    return,
        X, 2-d numpy.arrays
        y, 1-d numpy.arrays
    """
    if isinstance(X, list): X = np.array(X)
    if isinstance(y, list): y = np.array(y)

    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)

    n_samples, n_features = X.shape
    y_samples = y.shape[0]

    if X.ndim != 2:
        raise TypeError("Train data X should be a 2-d arrays.")

    if n_samples != y_samples:
        raise Exception(
            "The number of samples is not fitted between X and y. "
            "Model can't train ({X_samples}, {y_samples})."
            .format(**dict(X_samples=n_samples, y_samples=y_samples))
        )

    return X, y


def check_predict_data(model, X):
    """Check type of data used to predict. This method accept
    `list` type.

    X_samples, X_features, X.shape
        Parameter `n_features` should fit to X in train period.
    X, 2-d numpy array
    """
    n_features = getattr(model, "n_features", None)

    if not hasattr(model, "train") or n_features is None:
        raise TypeError("%s is not an model instance." % model)

    if isinstance(X, list): X = np.array(X)

    assert isinstance(X, np.ndarray)

    if X.ndim == 1:
        X = np.array([list(X)])

    X_samples, X_features = X.shape

    if n_features != X_features:
        raise Exception(
            "The model can't predict (,{X_features}) sample."
            .format(**dict(X_features=X_features))
        )

    return X


def check_two_arrays(x, y):
    """check two arrays.

    x, 1-d numpy.arrays
    y, 1-d numpy.arrays
    """
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)

    if len(x) != len(y):
        raise Exception("Two sample should have same length.")
