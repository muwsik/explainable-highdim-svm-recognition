import numpy as np

from sample import Sample


def generateSample(
    nSamples,          # number of objects of both classes
    nFeatures,         # total number of features
    #nInformative,      # informative features
    a = None,          # direction vector of separating hyperplane
    b = 0.0,           # offset along the direction vector
    scale = 1.0,
    seed = None
):
    rng = np.random.default_rng(seed)

    # 
    X = rng.standard_normal((nSamples, nFeatures)) * scale

    #
    a /= np.linalg.norm(a)

    # 
    scores = np.dot(X, a) - b

    # noise

    # lables
    Y = np.where(scores > 0, 1, -1)

    return Sample(
        X,
        Y,
        {
            "nSamples": nSamples,
            "nFeatures": nFeatures,
            #"nInformative": nInformative,
            "a": a,
            "b": b,
            "scale": scale,
            "seed": seed
        }
    )

if __name__ == "__main__":
    temp_a = np.random.uniform(low = 0, high = 1, size = 1000)

    trainDataset = generateSample(
        nSamples = 10000,
        nFeatures = 1000,
        #nInformative = 1000,
        a = temp_a,
        b = 0,
        scale = 1
    )
    #trainDataset.saveBin(r"D:\datasets\train-10k-1k.npz")

    testDataset = generateSample(
        nSamples = 10000,
        nFeatures = 1000,
        #nInformative = 1000,
        a = temp_a,
        b = 0,
        scale = 1
    )
    #testDataset.saveBin(r"D:\datasets\test-10k-1k.npz")