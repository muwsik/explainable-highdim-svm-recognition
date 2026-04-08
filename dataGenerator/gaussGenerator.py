import numpy as np

from sample import Sample


def generateSample(
    nSamples,          # number of objects of both classes
    nFeatures,         # total number of features
    nInformative,      # informative features
    a = None,          # direction vector of separating hyperplane
    b = 0.0,           # offset along the direction vector
    scale = 1.0,
    seed = None
):
    if (nInformative > nFeatures):
        raise  ValueError("!")

    rng = np.random.default_rng(seed)

    # 
    X = rng.standard_normal((nSamples, nInformative)) * scale

    # 
    if a is None:
        a = np.zeros(nFeatures)
        temp = rng.standard_normal(nInformative)
        temp /= np.linalg.norm(temp)
        a[:nInformative] = temp

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
            "nInformative": nInformative,
            "a": a,
            "b": b,
            "scale": scale,
            "seed": seed
        }
    )