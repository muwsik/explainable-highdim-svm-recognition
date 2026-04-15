from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import numpy as np

from dataGenerator.sample import Sample

# параметры
n_samples = 20000
n_features = 5000
n_informative = 2500
n_redundantint = 2500

# 
X, y = make_classification(
    n_samples = n_samples,
    n_features = n_features,
    n_informative = n_informative,
    n_redundant = n_redundantint,
    n_repeated = 0,
    n_classes = 2,
    n_clusters_per_class = 1,
    class_sep = 1.0,
    # random_state = 42
)

y =  np.where(y > 0, 1, -1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.5
)

namePart = f"{n_samples//2000}k-f{n_features}-i{n_informative}-r{n_redundantint}-l{n_features - n_informative - n_redundantint}"

trainSample = Sample(X_train, y_train, {"a": np.zeros(n_features)})
trainSample.saveBin(rf"D:\datasets\mc-train-2-{namePart}.npz")

testSample = Sample(X_test, y_test, {"a": np.zeros(n_features)})
testSample.saveBin(rf"D:\datasets\mc-test-2-{namePart}.npz")