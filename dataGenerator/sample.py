import numpy as np
import os


# Organization of sample data storage
class Sample:
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