import numpy as np
import time

from sklearn.svm import SVC


class combLinModel:
    def __init__(self, numIndex, numSplit, seed = None):
        if seed is None:            
            inds = np.random.permutation(numIndex)
        else:
            rng = np.random.default_rng(seed)
            inds = rng.permutation(numIndex)

        self.subspaceIndex = np.array_split(inds, numSplit)
        self.subspaceModels = None 


    def fit(self, X, Y, baseModelCreater = lambda: SVC(kernel = 'linear')):
        self.subspaceModels = []

        # training by subspaces
        for tempSubspace in self.subspaceIndex:
            timeStartTrain = time.time()
            tempModel = baseModelCreater()
            tempModel.fit(X[:, tempSubspace], Y)
            timeEndTrain = time.time()

            self.subspaceModels.append([tempModel, timeEndTrain - timeStartTrain])

        # combining general solution 
        self.a = []
        self.b = 0
        for tempModel, _ in self.subspaceModels:
            self.a = np.hstack((self.a, tempModel.coef_[0]))
            self.b += tempModel.intercept_

        self.b /= len(self.subspaceIndex)


    def decision_function(self, X):
        inds = np.concatenate(self.subspaceIndex)
        scores = np.dot(X[:, inds], self.a) + self.b
        return scores


    def predict(self, X):
        scores = self.decision_function(X)
        labels = np.where(scores >= 0, 1, -1)
        return labels