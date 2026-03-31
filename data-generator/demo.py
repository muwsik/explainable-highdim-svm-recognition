from tkinter import OFF
import numpy as np
import matplotlib.pyplot as plt

import generator as gen

# 2D visualization
def plot2D(tempLinearSample, vectorA, offset):
    pos = tempLinearSample.X[tempLinearSample.Y ==  1]
    neg = tempLinearSample.X[tempLinearSample.Y == -1]

    plt.figure(figsize = (6, 6))

    plt.scatter(pos[:, 0], pos[:, 1], c = "blue", label = "+1", alpha = 0.6)
    plt.scatter(neg[:, 0], neg[:, 1], c = "red", label = "-1", alpha = 0.6)

    # separating rule
    vectorA = np.array(vectorA, dtype = float)
    xmin, xmax = np.min(tempLinearSample.X[:, 0]), np.max(tempLinearSample.X[:, 0])
    ymin, ymax = np.min(tempLinearSample.X[:, 1]), np.max(tempLinearSample.X[:, 1])
    if abs(vectorA[1]) >= 1e-6:
        xLine = np.linspace(xmin, xmax, 2)
        yLine = -(vectorA[0] * xLine - offset)/ vectorA[1]

        plt.plot(xLine, yLine, "k-", label = "rule")
        plt.plot(xLine - 1, yLine - 1, "k--", label = "border")
        plt.plot(xLine + 1, yLine + 1, "k--")
    else:
        x0 = offset / vectorA[0]
        plt.axvline(x0, color = "black", linestyle = "-", label = "rule")
        plt.axvline(x0 + 1 / vectorA[0], color = "black", linestyle = "--", label = "margin")
        plt.axvline(x0 - 1 / vectorA[0], color = "black", linestyle = "--")

    plt.xlim(xmin - 1, xmax + 1)
    plt.ylim(ymin - 1, ymax + 1)   

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


# 3D visualization
def drawPlane(a, c, xmin, xmax, ymin, ymax, zmin, zmax, alpha, color):    
    if abs(a[0]) >= 1e-6:
        Y, Z = np.meshgrid(
            np.linspace(ymin, ymax, 10),
            np.linspace(zmin, zmax, 10)
        )
        X = (c - a[1]*Y - a[2]*Z) / a[0]

    elif abs(a[1]) >= 1e-6:
        X, Z = np.meshgrid(
            np.linspace(xmin, xmax, 10),
            np.linspace(zmin, zmax, 10)
        )
        Y = (c - a[0]*X - a[2]*Z) / a[1]

    else:
        X, Y = np.meshgrid(
            np.linspace(xmin, xmax, 10),
            np.linspace(ymin, ymax, 10)
        )
        Z = (c - a[0]*X - a[1]*Y) / a[2]

    plt.gca().plot_surface(
        X, Y, Z,
        alpha = alpha,
        color = color
    )

def plot3D(tempLinearSample, vectorA, offset):
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
    
    drawPlane(vectorA, offset, xmin, xmax, ymin, ymax, zmin, zmax, 0.3, "green")
    # drawPlane(vectorA, -1 + offset, xmin, xmax, ymin, ymax, zmin, zmax, 0.1, "gray")
    # drawPlane(vectorA, +1 + offset, xmin, xmax, ymin, ymax, zmin, zmax, 0.1, "gray")

    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)
    ax.set_zlim(zmin - 1, zmax + 1)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend()

    
# DEMO
tempSeed = None

if __name__ == "__main__":
    objNum = 100
    halfSize = 10
    sigma = 1

    # # 2D generation and visualization
    # linGenerator = gen.LinearGenerator(tempSeed)
    # baseLinearSample = linGenerator.base(
    #     objNum,
    #     2,
    #     halfSize,
    #     sigma
    # )   

    # offset2D = 3
    # customLinearSample, a2D = gen.LinearGenerator(tempSeed).specifiedHyperplane(
    #     objNum, 
    #     2, 
    #     halfSize, 
    #     sigma, 
    #     vectorA = [1, 0],
    #     offset = offset2D
    # )

    # plot2D(baseLinearSample, [1, 0], 0)
    # plt.title(f"a = [1, 0], b = 0")
    # plot2D(customLinearSample, a2D, offset2D)
    # plt.title(f"a = {a2D}, b = {offset2D}")
    # plt.show()

    # 3D generation and visualization
    linGenerator = gen.LinearGenerator(tempSeed)
    baseLinearSample = linGenerator.base(
        objNum,
        3,
        halfSize,
        sigma
    )
    
    offset3D = -5
    customLinearSample, a3D = gen.LinearGenerator(tempSeed).specifiedHyperplane(
        objNum, 
        3, 
        halfSize, 
        sigma, 
        vectorA = [1, 0, 0],
        offset = offset3D
    )
    
    plot3D(baseLinearSample, [1, 0, 0], 0)
    plt.title(f"a = [1, 0, 0], b = 0")
    plot3D(customLinearSample, a3D, offset3D)
    plt.title(f"a = {a3D}, b = {offset3D}")
    plt.show()


    ## TXT save and load
    # tempSample = gen.LinearGenerator(tempSeed).base(
    #     objNum,
    #     3,
    #     halfSize,
    #     sigma
    # )
    # tempSample.saveTXT(r'D:\tempSample.txt')
    
    # sampleFromFile = gen.LinearSample()
    # sampleFromFile.loadTXT(r'D:\tempSample.txt')


    ## binary save and load
    # tempSample = gen.LinearGenerator(tempSeed).base(
    #     objNum,
    #     3,
    #     halfSize,
    #     sigma
    # )
    # tempSample.saveBin(r'D:\tempSample.npz')
    
    # sampleFromFile = gen.LinearSample()
    # sampleFromFile.loadBin(r'D:\tempSample.npz')
    
    pass