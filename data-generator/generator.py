import numpy as np
import os

"""
Generating data based on the model from the article (3.2.1)
    N: objects number of same class
    p: the number of features
    a: side size of the hypercube
    c: exponent parameter
    seed: initial state of the random number generator
Return
    X: object-feature matrix
    Y: class of an object in object-feature matrix
"""
def generateLinearSample(N, p, a, c, seed = None):   
    if seed is not None:
        np.random.seed(seed)

    # objects number of same class for uniform distribution
    n_left = abs(a * N / (a + (1 / c) * (1 - np.exp(-c * a))))

    if n_left > int(n_left):
        n_left = int(n_left) + 1
    else:
        n_left = int(n_left)

    # objects number of same class for exponential distribution
    n_right = N - n_left

    # first feature positive class
    pos_exp_obj = np.random.exponential(scale = 1/c, size = n_right) / c # .. / c  why?
    pos_norm_obj = np.random.uniform(-a, 0, size = n_left)

    # first feature negative class
    neg_exp_obj = -1 * np.random.exponential(scale = 1/c, size = n_right) / c   # .. / c  why?
    neg_norm_obj = -1 * np.random.uniform(-a, 0, size = n_left)

    # first feature
    first_feature = np.concatenate([
        np.concatenate([pos_exp_obj, pos_norm_obj]), 
        np.concatenate([neg_exp_obj, neg_norm_obj])
    ])

    # other features
    other_features = np.random.uniform(-a, a, size=(2 * N, p - 1))

    # object-feature matrix
    X = np.column_stack([first_feature, other_features])
    Y = np.concatenate([np.ones(N), -1 * np.ones(N)])

    return X, Y

""""""
def saveSample(filename, X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array (object-feature matrix)")

    if Y.ndim != 1:
        raise ValueError("Y must be a 1D array")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Number of objects mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
        )

    data = np.column_stack((Y, X))

    try:
        np.savetxt(filename, data, delimiter=" ", fmt="%.6f")
    except IOError as e:
        raise IOError(f"Cannot write file {filename}: {e}")

""""""
def loadSample(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist")

    try:
        data = np.loadtxt(filename, delimiter=" ")
    except Exception as e:
        raise ValueError(f"Error reading file '{filename}': {e}")

    if data.size == 0:
        raise ValueError("File is empty")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError("File must contain at least two columns: y and x1")

    Y = data[:, 0]
    X = data[:, 1:]

    unique_classes = np.unique(Y)
    if not np.all(np.isin(unique_classes, [-1, 1])):
        raise ValueError(f"Unexpected class labels: {unique_classes}")

    return X, Y