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

        # combining general solution as an average of subspace particular models
        for tempModel, _ in self.subspaceModels:
            # normalization of the subspace model to preserve scale
            tempNorm = np.linalg.norm(tempModel.coef_[0])
            tempNormA = tempModel.coef_[0] / tempNorm
            tempNormB = tempModel.intercept_[0] / tempNorm

            # since the models are built in orthogonal coordinate systems,
            # their addition in the final expanded space can be replaced by a simple union
            self.a = np.hstack((self.a, tempNormA))

            # displacement is added by the property of linear functions
            self.b += tempNormB

        # statistical averaging of models -> 1 / √N
        # algebraic averaging of models -> 1 / N
        self.a /= np.sqrt(len(self.subspaceIndex))
        self.b /= np.sqrt(len(self.subspaceIndex))

        # initial order of features
        temp = np.empty_like(self.a)
        temp[np.concatenate(self.subspaceIndex)] = self.a
        self.coef_ = [temp]

        # 
        self.intercept_ = [self.b]


    def decision_function(self, X):
        inds = np.concatenate(self.subspaceIndex)
        scores = np.dot(X[:, inds], self.a) + self.b
        return scores


    def predict(self, X):
        scores = self.decision_function(X)
        labels = np.where(scores >= 0, 1, -1)
        return labels


    def recalucateIntercept(self, X, Y, eps = 1e-2):        
        inds = np.concatenate(self.subspaceIndex)
        decision  = np.dot(X[:, inds], self.a)

        mask = np.abs(Y*decision - 1) < eps

        b = None
        if (np.sum(mask) > 0):
            b = np.mean(Y[mask] - decision[mask])

        self.b = b
        return b