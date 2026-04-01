import numpy as np

class combModel:
    def __init__(self, numIndex, numSplit, seed = None):
        if seed is None:            
            inds = np.random.permutation(numIndex)
        else:
            rng = np.random.default_rng(seed)
            inds = rng.permutation(numIndex)

        self.subspaceIndex = np.array_split(inds, numSplit)


    def fit(self, X, Y, basemModel):
        pass


    def subspacePredict(self, X):
        pass


    def predict(self, X):
        pass