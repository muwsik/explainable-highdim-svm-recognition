import numpy as np
import time

from sklearn.svm import SVC


class combLinModel:
    def __init__(self, numSplits, baseModel = lambda: SVC(kernel = 'linear'), seed = None):
        self.numSplits = numSplits
        self.baseModel = baseModel

        self.generator = np.random.default_rng(seed)

        self.subspaceIndex = None
        self.subspaceModels = None         
        self.a = np.array([])
        self.b = 0


    def fit(self, X, Y):
        # slit fearutes on subspaces   
        numIndex = X.shape[1]     
        inds = self.generator.permutation(numIndex)
        self.subspaceIndex = np.array_split(inds, self.numSplits)
        self.subspaceModels = []

        # training by subspaces
        for tempSubspace in self.subspaceIndex:
            timeStartTrain = time.time()
            tempModel = self.baseModel()
            tempModel.fit(X[:, tempSubspace], Y)
            timeEndTrain = time.time()

            self.subspaceModels.append([tempModel, timeEndTrain - timeStartTrain])

        # combining general solution 
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