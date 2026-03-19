import numpy as np
import os

import matplotlib.pyplot as plt

# Organization of sample data storage
class LinearSample:
    def __init__(self, _X = None, _Y = None):
        self.X = _X
        self.Y = _Y

    def Save(self, filename, delim = ' '):
        if (self.X is None) or (self.Y is None):
            raise ValueError("Incorrect data for saving")

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array (object-feature matrix)")

        if self.Y.ndim != 1:
            raise ValueError("Y must be a 1D array")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"Number of objects mismatch: X has {self.X.shape[0]}, Y has {self.Y.shape[0]}"
            )

        data = np.column_stack((self.Y, self.X))

        try:
            np.savetxt(filename, data, delimiter = delim, fmt = "%.6f")
        except IOError as e:
            raise IOError(f"Cannot write file {filename}: {e}")

    def Load(self, filename, delim = ' '):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist")

        try:
            data = np.loadtxt(filename, delimiter = delim)
        except Exception as e:
            raise ValueError(f"Error reading file '{filename}': {e}")

        if data.size == 0:
            raise ValueError("File is empty")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 2:
            raise ValueError("File must contain at least two columns: y and x1")

        self.Y = data[:, 0]
        self.X = data[:, 1:]

        unique_classes = np.unique(self.Y)
        if not np.all(np.isin(unique_classes, [-1, 1])):
            raise ValueError(f"Unexpected class labels: {unique_classes}")


# Generating different variants of linear samples
class LinearGenerator:
    def __init__(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
    
    def base(self, objNum, featNum, halfSize, sigma): 
        """
        Generating data based on the linear model with direction vector a = [1,0,0,0,...,0]
            objNum: objects number of same class
            featNum: the number of features
            halfSize: half edge size of hypercube
            sigma: exponent parameter
        Return 
            LinearSample.X: object-feature matrix
            LinearSample.Y: class of an object in object-feature matrix
        """  

        # objects number of same class for uniform distribution
        n_left = abs(
            halfSize * objNum / (
                halfSize + (1 / sigma) * (1 - np.exp(-sigma * halfSize))
                )
            )
        
        # objects number of same class for uniform distribution
        n_left = int(n_left)

        # objects number of same class for exponential distribution
        n_right = objNum - n_left

        # first feature positive class
        pos_exp_obj = np.random.exponential(scale = 1/sigma, size = n_right)
        pos_norm_obj = np.random.uniform(-halfSize, 0, size = n_left)

        # first feature negative class
        neg_exp_obj = -1 * np.random.exponential(scale = 1/sigma, size = n_right)
        neg_norm_obj = -1 * np.random.uniform(-halfSize, 0, size = n_left)

        # first feature
        first_feature = np.concatenate([
            np.concatenate([pos_exp_obj, pos_norm_obj]), 
            np.concatenate([neg_exp_obj, neg_norm_obj])
        ])

        # other features with uniform distribution
        other_features = np.random.uniform(-halfSize, halfSize, size=(2 * objNum, featNum - 1))

        # object-feature matrix
        X = np.column_stack([first_feature, other_features])
        Y = np.concatenate([np.ones(objNum), -1 * np.ones(objNum)])

        return LinearSample(X, Y)

    def specifiedHyperplane(self, objNum, featNum, halfSize, sigma, vectorA = None): 
        """
        Generating data based on the linear model with hyperplane custom direction.
        If parametr a is None, then it will be random.
            objNum: objects number of same class
            featNum: the number of features
            halfSize: half edge size of hypercube
            sigma: exponent parameter
            a: custom direction vector by hyperplane
        Return
            LinearSample.X: object-feature matrix
            LinearSample.Y: class of an object in object-feature matrix
            vectorA: normalized directing vector of hyperplane
        """

        if (vectorA is None):
            vectorA = np.random.normal(size = featNum)

        vectorA = np.array(vectorA, dtype = float)
        if len(vectorA) != featNum:
            raise ValueError("len(a) must be featNum")   

        # construct an orthonormal basis by Householder  
        vectorA = vectorA / np.linalg.norm(vectorA)
        e1 = np.zeros(featNum)
        e1[0] = 1.0 # because we will use the data for the case a = [1,0,0,0,...,0]
        v = e1 - vectorA
        normV = np.linalg.norm(v)
        
        H = np.eye(featNum)
        if normV >= 1e-6:
            v = v / normV
            H = H - 2 * np.outer(v, v)            
            
        # generate base linear model
        baseSample = self.base(objNum, featNum, halfSize, sigma)

        # turning to an orthonormal basis
        rotationX = np.dot(baseSample.X, H)

        return LinearSample(rotationX, baseSample.Y), vectorA