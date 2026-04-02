import numpy as np
import time


class combModel:
    def __init__(self, numIndex, numSplit, seed = None):
        if seed is None:            
            inds = np.random.permutation(numIndex)
        else:
            rng = np.random.default_rng(seed)
            inds = rng.permutation(numIndex)

        self.subspaceIndex = np.array_split(inds, numSplit)    


    def fit(self, X, Y, baseModelCreater):
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


    def predict(self, X):
        inds = np.concatenate(self.subspaceIndex)

        scores = np.dot(X[:, inds], self.a) + self.b

        labels = np.where(scores >= 0, 1, -1)

        return scores, labels


    def subspacePredict(self, X):
        pass