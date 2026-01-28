from math import log

import numpy as np
import pandas as pd

__author__ = "Victor Ruiz, vmr11@pitt.edu"


def compute_entropy(y: pd.Series | np.ndarray, base: int = 2) -> float:
    """
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param y: Series with labels of examples in a dataset
    :return: value of entropy
    """
    assert len(y.shape) == 1, f"compute_entropy - {y.shape=}"

    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    classes = y.unique()
    n_classes = len(y)
    ent = 0.0  # initialize entropy

    # iterate over classes
    for c in classes:
        partition = y[y == c]  # data with class = c
        proportion = 1.0 * len(partition) / n_classes
        # update entropy
        ent -= proportion * log(proportion, base)

    return ent


def cut_point_information_gain(
    X: pd.DataFrame | np.ndarray,
    cut_point: float,
    feature_label: int | str,
    class_label: int | str,
) -> float:
    """
    Returns the information gain obtained by splitting a numeric attribute in two according to cut_point
    :param X: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    """
    assert len(X.shape) == 2, f"cut_point_information_gain - {X.shape=}"

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    entropy_full = compute_entropy(
        X[class_label]
    )  # compute entropy of full dataset (w/o split)

    # split data at cut_point
    data_left = X[X[feature_label] <= cut_point]
    data_right = X[X[feature_label] > cut_point]
    (N, N_left, N_right) = (len(X), len(data_left), len(data_right))

    gain = (
        entropy_full
        - (N_left / N) * compute_entropy(data_left[class_label])
        - (N_right / N) * compute_entropy(data_right[class_label])
    )

    return gain
