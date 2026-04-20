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
            self.b += tempModel.intercept_[0]

        self.a /= np.linalg.norm(self.a) 
        self.b /= len(self.subspaceIndex)

        #
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
    

def computeIntercept(model, X, Y, eps = 1e-3):
    """
    Estimate the bias *b* ONLY for LinearSVC using points near the margin.
        model: trained LinearSVC model
        X: objects of training sample
        Y: objects labels in {-1, 1}
        eps: tolerance for selecting points with *margin = 1 ± eps*
    Return
        b: estimated hyperplane displacement
    """

    # 
    decision = model.decision_function(X)
    margin = Y * decision

    # potentially supporting objects
    mask = np.abs(margin - 1) < eps
    #print(f"count potentially supporting objects: {np.sum(mask)}")

    b = None
    if (np.sum(mask) > 0):
        b = np.mean(Y[mask] - np.dot(X[mask], model.coef_[0]))
    # else no potentially supporting objects with tolerance *eps*
    # you can increase the value *eps*
    
    return b


if __name__ == "__main__":
    from sklearn.svm import LinearSVC
    from dataGenerator.sample import Sample

    trainDataset = Sample.fromBin(r"D:\datasets\b-10k-1k.npz")

    model = LinearSVC(penalty = 'l1', dual = False, verbose = True)
    model.fit(trainDataset.X, trainDataset.Y)

    b = computeIntercept(model, trainDataset.X, trainDataset.Y)
