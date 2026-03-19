import numpy as np
import matplotlib.pyplot as plt

import generator as gen

# 2D visualization
def plot2D(tempLinearSample, vectorA):
    pos = tempLinearSample.X[tempLinearSample.Y ==  1]
    neg = tempLinearSample.X[tempLinearSample.Y == -1]

    plt.figure(figsize = (6, 6))

    plt.scatter(pos[:, 0], pos[:, 1], c = "blue", label = "+1", alpha = 0.6)
    plt.scatter(neg[:, 0], neg[:, 1], c = "red", label = "-1", alpha = 0.6)
    plt.scatter(neg[5, 0], neg[5, 1], c = "green", label = "-1", s = 100)

    # separating rule
    vectorA = np.array(vectorA, dtype = float)
    xmin, xmax = np.min(tempLinearSample.X[:, 0]), np.max(tempLinearSample.X[:, 0])
    ymin, ymax = np.min(tempLinearSample.X[:, 1]), np.max(tempLinearSample.X[:, 1])
    if abs(vectorA[1]) >= 1e-6:
        x = np.linspace(xmin, xmax, 3)
        plt.plot(x, -(vectorA[0] / vectorA[1]) * x, "k--", label = "rule")
    else:
        plt.axvline(
            0,
            color = "black",
            linestyle = "--",
            label = "rule"
        )

    plt.xlim(xmin - 1, xmax + 1)
    plt.ylim(ymin - 1, ymax + 1)   

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


# 3D visualization
def plot3D(tempLinearSample, vectorA):
    pos = tempLinearSample.X[tempLinearSample.Y ==  1]
    neg = tempLinearSample.X[tempLinearSample.Y == -1]

    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(111, projection = "3d")

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c = "blue", label = "+1", alpha = .6)

    ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2],
               c = "red", label = "-1", alpha = 0.6)

    # separating rule
    vectorA = np.array(vectorA, dtype = float)
    xmin, xmax = np.min(tempLinearSample.X[:, 0]), np.max(tempLinearSample.X[:, 0])
    ymin, ymax = np.min(tempLinearSample.X[:, 1]), np.max(tempLinearSample.X[:, 1])
    zmin, zmax = np.min(tempLinearSample.X[:, 2]), np.max(tempLinearSample.X[:, 2])
    if abs(vectorA[0]) >= 1e-6:
        Y, Z = np.meshgrid(
            np.linspace(ymin, ymax, 10),
            np.linspace(zmin, zmax, 10)
        )
        X = -(vectorA[1] * Y + vectorA[2] * Z) / vectorA[0]
    elif abs(vectorA[1]) >= 1e-6:
        X, Z = np.meshgrid(
            np.linspace(xmin, xmax, 10),
            np.linspace(zmin, zmax, 10)
        )
        Y = -(vectorA[0] * X + vectorA[2] * Z) / vectorA[1]

    else:
        X, Y = np.meshgrid(
            np.linspace(xmin, xmax, 10),
            np.linspace(ymin, ymax, 10)
        )
        Z = -(vectorA[0] * X + vectorA[1] * Y) / vectorA[2]

    ax.plot_surface(
        X, Y, Z,
        alpha = 0.3,
        color = "gray"
    )

    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)
    ax.set_zlim(zmin - 1, zmax + 1)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend()

    
# DEMO
tempSeed = 1

if __name__ == "__main__":
    objNum = 10
    halfSize = 10
    sigma = 0.75

    # 2D case
    linGenerator = gen.LinearGenerator(tempSeed)
    baseLinearSample = linGenerator.base(
        objNum,
        2,
        halfSize,
        sigma
    )   

    suctomLinearSample, a2D = gen.LinearGenerator(tempSeed).specifiedHyperplane(
        objNum, 
        2, 
        halfSize, 
        sigma, 
        vectorA = [1, 1]
    )

    plot2D(baseLinearSample, [1, 0])
    plt.title(f"a = [1, 0]")
    plot2D(suctomLinearSample, a2D)
    plt.title(f"a = {a2D}")

    #3D case
    linGenerator = gen.LinearGenerator(tempSeed)
    baseLinearSample = linGenerator.base(
        objNum,
        3,
        halfSize,
        sigma
    )

    suctomLinearSample, a3D = gen.LinearGenerator(tempSeed).specifiedHyperplane(
        objNum, 
        3, 
        halfSize, 
        sigma, 
        vectorA = [1, 1, 1]
    )
    
    plot3D(baseLinearSample, [1, 0, 0])
    plt.title(f"a = [1, 0, 0]")
    plot3D(suctomLinearSample, a3D)
    plt.title(f"a = {a3D}")

    plt.show()