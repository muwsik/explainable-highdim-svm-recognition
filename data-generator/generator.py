import numpy as np

"""
Generating data based on the model from the article (3.2.1)
    N   - objects number of same class
    p   - the number of features
    a   -  side size of the hypercube
    c   - exponent parameter
    seed    - initial state of the random number generator
Return
    X   - object-feature matrix
    Y   - class of an object in object-feature matrix

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