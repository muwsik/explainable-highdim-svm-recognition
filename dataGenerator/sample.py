import numpy as np
import os

from sklearn import datasets


# Organization of sample data storage
class Sample:
    def __init__(self, _X = None, _Y = None):
        self.X = _X
        self.Y = _Y

    # verification
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


    # (1) for text data in format "<label> feature1 feature2 ... featureN"
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


    # (2) for dense binary data
    def saveBin(self, filename):
        self.check()
        np.savez_compressed(filename,
            X = self.X,
            Y = self.Y
        )

    def loadBin(self, filename):        
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        
        data = np.load(filename, allow_pickle = True)

        if "X" not in data or "Y" not in data:
            raise ValueError("Invalid file")

        self.X = data["X"]
        self.Y = data["Y"]
        
        self.check()


    # (3) for sparse text datа in format libsvm
    def loadSparse(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        X, Y = datasets.load_svmlight_file(filename)

        self.X = X 
        self.Y = Y.astype(np.int8)

        self.check()

    def saveSparse(self, filename):
        self.check()
        datasets.dump_svmlight_file(
            self.X,
            self.Y.astype(np.float64),
            filename,
            zero_based = False
        )


    # automatic uploading data from a file in one of the supported formats
    @classmethod
    def fromFile(cls, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        
        obj = cls()        
        
        try: # dense .npz (2)
            obj.loadBin(filename)
            return obj 
        except Exception:
            pass

        try: # sparse .txt (3)
            obj.loadSparse(filename)
            return obj 
        except Exception:
            pass
        
        try: # dense .txt (1)
            obj.loadTXT(filename)
            return obj 
        except Exception:
            pass
        
        raise ValueError(f"Unknown or unsupported file format: {filename}")
    
if __name__ == "__main__":
    
    tempPath = r"D:\Cloud\SVM\dataset\synthetic12-25k-f2500-i1500-r500-l500.npz"

    dataset = Sample.fromFile(tempPath)
    print(f"Loded dataset X:{dataset.X.shape} Y:{dataset.Y.shape}")
    
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    model = LinearSVC(C = 10, penalty = 'l1', dual = False, verbose = True)

    model.fit(dataset.X, dataset.Y)

    print(accuracy_score(dataset.Y, model.predict(dataset.X)))