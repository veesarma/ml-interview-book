"""Classical machine learning, implemented from scratch in NumPy.

Every function documents its input / output shapes with the book's dimension
vocabulary: ``N`` examples, ``d`` features, ``K`` classes, ``k`` clusters /
components, ``M`` trees / boosting rounds.
"""

from . import (
    decision_tree,
    gda,
    gmm,
    gradient_boosting,
    kernels,
    kmeans,
    knn,
    linear_regression,
    logistic_regression,
    naive_bayes,
    pca,
    random_forest,
    softmax_regression,
    svm,
)

__all__ = [
    "linear_regression",
    "logistic_regression",
    "softmax_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "knn",
    "kmeans",
    "naive_bayes",
    "gda",
    "gmm",
    "pca",
    "kernels",
    "svm",
]
