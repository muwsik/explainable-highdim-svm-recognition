from scipy import sparse
import numpy as np
import time

from sklearn.svm import LinearSVC

from joblib import Parallel, delayed

class combBinModel:
    def __init__(self,
        numSplits, 
        baseModel = lambda: LinearSVC(C = 1, penalty = 'l1', dual = False),
        seed = None,
        nJobs = -1
    ):
        self.numSplits = numSplits
        self.baseModel = baseModel

        self.nJobs = nJobs
        self.generator = np.random.default_rng(seed)
        self.subspaceIndex = None
        self.subspaceModels = None 

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

    def fit(self, X, Y):
        w = []      # weight vector (normal vector hyperplane)
        b = 0       # bias term hyperplane

        # slit fearutes on subspaces   
        numIndex = X.shape[1]     
        shuffledIndexes = self.generator.permutation(numIndex)
        self.subspaceIndex = np.array_split(shuffledIndexes, self.numSplits)
        self.subspaceModels = []

        # training by subspaces 
        def trainSubspace(tempSubspace):
            timeStartTrain = time.time()
            tempModel = self.baseModel()
            tempModel.fit(X[:, tempSubspace], Y)
            timeEndTrain = time.time()

            return (tempModel, timeEndTrain - timeStartTrain)

        parallelExecutor = Parallel(n_jobs = self.nJobs, backend = "loky")        
        self.subspaceModels = parallelExecutor(
            delayed(trainSubspace)(tempSubspace) for tempSubspace in self.subspaceIndex
        )

        # combining general solution as an average of subspace particular models
        for tempModel, _ in self.subspaceModels:
            # normalization of the subspace model to preserve scale
            tempNorm = np.linalg.norm(tempModel.coef_[0])
            if (tempNorm == 0):
                tempNormA = np.zeros_like(tempModel.coef_[0])
                tempNormB = 0.0
            else:
                tempNormA = tempModel.coef_[0] / tempNorm
                tempNormB = tempModel.intercept_[0] / tempNorm

            # since the models are built in orthogonal coordinate systems,
            # their addition in the final expanded space can be replaced by a simple union
            w.append(tempNormA)

            # bias is added by property of linear functions
            b += tempNormB

        # statistical averaging of models -> 1 / √N
        # algebraic averaging of models -> 1 / N
        w = np.concatenate(w) / np.sqrt(len(self.subspaceIndex))
        b /= np.sqrt(len(self.subspaceIndex))

        # initial order of features
        temp = np.empty_like(w)
        temp[shuffledIndexes] = w
        self.coef_ = [temp]

        # bias
        self.intercept_ = [b]

        #
        self.n_iter_ = np.median([i[0].n_iter_ for i in self.subspaceModels])


    def decision_function(self, X):
        w = self.coef_[0]
        
        if sparse.issparse(X):
            scores = X.dot(w) + self.intercept_[0]
        else:
            scores = np.dot(X, w) + self.intercept_[0]

        return scores


    def predict(self, X):
        scores = self.decision_function(X)
        labels = np.where(scores >= 0, 1, -1)
        return labels
