import numpy as np
import os


# Organization of sample data storage
class LinearSample:
    def __init__(self, _X = None, _Y = None, _params = None):
        self.X = _X
        self.Y = _Y
        self.params = _params


    def check(self):
        if self.X is None or self.Y is None:
            raise ValueError("No data")

        if self.X.ndim != 2:
            raise ValueError("X must be 2D")

        if self.Y.ndim != 1:
            raise ValueError("Y must be 1D")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y size mismatch")

        u = np.unique(self.Y)
        if not np.all(np.isin(u, [-1, 1])):
            raise ValueError(f"Unexpected classes: {u}")


    def saveTXT(self, filename, append = True, delim = ' '):
        self.check()

        data = np.column_stack((self.Y, self.X))

        with open(filename, "a" if append else "w") as tempF:
            np.savetxt(tempF, data, delimiter = delim, fmt = "%.3f")


    def loadTXT(self, filename, delim = ' '):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        data = np.loadtxt(filename, delimiter = delim, dtype = np.float32)

        self.Y = data[:, 0].astype(np.int8)
        self.X = data[:, 1:]

        self.check()


    def saveBin(self, filename):
        self.check()

        np.savez_compressed(filename,
            X = self.X,
            Y = self.Y,
            params = self.params
        )


    def loadBin(self, filename):
        data = np.load(filename, allow_pickle = True)

        if "X" not in data or "Y" not in data:
            raise ValueError("Invalid file")

        self.X = data["X"]
        self.Y = data["Y"]
        
        if "params" in data:
            self.params = data["params"].item()
        else:
            self.params = {}

        self.check()



# Generating different variants of linear samples
class LinearGenerator:
    def __init__(self, seed = None):
        if seed is not None:
            np.random.seed(seed)    


    # generating a truncated exponential distribution
    @staticmethod
    def __truncExp(n, sigma, length):        
        u = np.random.uniform(0, 1, n).astype(np.float32)
        return (
            -np.log(
                1 - u * (1 - np.exp(-sigma * length))
            ) / sigma
        ).astype(np.float32)


    def base(self, objNum, featNum, halfSize, sigma): 
        """
        Generating data based on the linear model with direction vector *a* = [1,0,0,0,...,0]
            objNum: objects number of same class
            featNum: the number of features
            halfSize: half edge size of hypercube
            sigma: exponent parameter
        Return 
            LinearSample.X: object-feature matrix
            LinearSample.Y: class of an object in object-feature matrix
            LinearSample.params: generation parameters
        """  
        
        # objects number of same class for exponential distribution
        numberExpDistPoints = int(
            ( objNum * (1 - np.exp(-sigma * (halfSize + 1))) )
            /
            ( sigma * (halfSize - 1) + (1 - np.exp(-sigma * (halfSize + 1))) ) 
        )

        # objects number of same class for uniform distribution
        numberUniformDistPoints = objNum - numberExpDistPoints

        # feature negative class in [-1, halfSize]
        negExp = -1 + LinearGenerator.__truncExp(numberExpDistPoints, sigma, halfSize + 1)
        # feature negative class in [-halfSize, -1]
        negUniform = np.random.uniform(-halfSize, -1, numberUniformDistPoints).astype(np.float32)

        # feature positive class in [-halfSize, 1]
        posExp = 1 - LinearGenerator.__truncExp(numberExpDistPoints, sigma, halfSize + 1)
        # feature positive class in [1, halfSize]
        posUniform = np.random.uniform(1, halfSize, numberUniformDistPoints).astype(np.float32)

        # first feature
        firstFeature = np.concatenate([
            np.concatenate([posExp, posUniform]), 
            np.concatenate([negExp, negUniform])
        ])

        # other features with uniform distribution
        otherFeatures = np.random.uniform(
            -halfSize,
            halfSize,
            size = (2 * objNum, featNum - 1)
            ).astype(np.float32)

        # object-feature matrix
        X = np.column_stack([firstFeature, otherFeatures])
        Y = np.concatenate([np.ones(objNum, dtype = np.int8), -1 * np.ones(objNum, dtype = np.int8)])

        return LinearSample(
            X,
            Y,
            {
                "objNum": objNum,
                "halfSize": halfSize,
                "featNum": featNum,
                "sigma": sigma,
                "a": np.concatenate([[1], np.zeros(featNum - 1)]),
                "b": 0
             }
        )


    def specifiedHyperplane(self, objNum, featNum, halfSize, sigma, a = None, b = None): 
        """
        Generating data based on the linear model with hyperplane custom direction.
        If parametr *a* is None, then it will be random.
            objNum: objects number of same class
            featNum: the number of features
            halfSize: half edge size of hypercube
            sigma: exponent parameter
            a: custom direction vector by hyperplane
            b: displacement of hyperplane along direction vector from origin
        Return
            LinearSample.X: object-feature matrix
            LinearSample.Y: class of an object in object-feature matrix
            a: normalized directing vector of hyperplane
        """

        if (a is None):
            a = np.random.normal(size = featNum).astype(np.float32)  

        a = np.array(a, dtype = np.float32)
        if len(a) != featNum:
            raise ValueError("len(a) must be featNum")   

        # construct an orthonormal basis by Householder  
        a = a / np.linalg.norm(a)
        e1 = np.zeros(featNum, dtype = np.float32)
        e1[0] = 1.0 # because we will use the sample with a = [1,0,0,0,...,0]
        v = e1 - a
        normV = np.linalg.norm(v)
        
        H = np.eye(featNum, dtype = np.float32)
        if normV >= 1e-6:
            v = v / normV
            H = H - 2 * np.outer(v, v).astype(np.float32)            
            
        # generate base linear model
        baseSample = self.base(objNum, featNum, halfSize, sigma)

        # turning to an orthonormal basis
        rotationX = np.dot(baseSample.X, H)

        if (b is not None):
            rotationX = rotationX + b * a

        return LinearSample(
            rotationX,
            baseSample.Y,
            {
                "objNum": objNum,
                "halfSize": halfSize,
                "featNum": featNum,
                "sigma": sigma,
                "a": a,
                "b": b
             }
        )