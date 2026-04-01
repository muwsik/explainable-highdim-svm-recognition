import numpy as np
import matplotlib.pyplot as plt

import linGenerator as gen

# 2D visualization
def plot2D(tempLinearSample):
    a = tempLinearSample.params["a"]
    b = tempLinearSample.params["b"]

    pos = tempLinearSample.X[tempLinearSample.Y ==  1]
    neg = tempLinearSample.X[tempLinearSample.Y == -1]

    plt.figure(figsize = (6, 6))

    plt.scatter(pos[:, 0], pos[:, 1], c = "blue", label = "+1", alpha = 0.6)
    plt.scatter(neg[:, 0], neg[:, 1], c = "red", label = "-1", alpha = 0.6)

    # separating rule
    a = np.array(a, dtype = float)
    xmin, xmax = np.min(tempLinearSample.X[:, 0]), np.max(tempLinearSample.X[:, 0])
    ymin, ymax = np.min(tempLinearSample.X[:, 1]), np.max(tempLinearSample.X[:, 1])
    if abs(a[1]) >= 1e-6:
        xLine = np.linspace(xmin, xmax, 2)
        yLine = -(a[0] * xLine - b)/ a[1]

        plt.plot(xLine, yLine, "k-", label = "rule")
        plt.plot(xLine - 1, yLine - 1, "k--", label = "border")
        plt.plot(xLine + 1, yLine + 1, "k--")
    else:
        x0 = b / a[0]
        plt.axvline(x0, color = "black", linestyle = "-", label = "rule")
        plt.axvline(x0 + 1 / a[0], color = "black", linestyle = "--", label = "margin")
        plt.axvline(x0 - 1 / a[0], color = "black", linestyle = "--")

    plt.xlim(xmin - 1, xmax + 1)
    plt.ylim(ymin - 1, ymax + 1)   

    plt.title(f"a = {a}, b = {b}")
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

def plot3D(tempLinearSample):
    a = tempLinearSample.params["a"]
    b = tempLinearSample.params["b"]

    pos = tempLinearSample.X[tempLinearSample.Y ==  1]
    neg = tempLinearSample.X[tempLinearSample.Y == -1]

    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(111, projection = "3d")

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c = "blue", label = "+1", alpha = 0.6)

    ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2],
               c = "red", label = "-1", alpha = 0.6)

    # separating rule
    a = np.array(a, dtype = float)
    xmin, xmax = np.min(tempLinearSample.X[:, 0]), np.max(tempLinearSample.X[:, 0])
    ymin, ymax = np.min(tempLinearSample.X[:, 1]), np.max(tempLinearSample.X[:, 1])
    zmin, zmax = np.min(tempLinearSample.X[:, 2]), np.max(tempLinearSample.X[:, 2])
    
    drawPlane(a, b, xmin, xmax, ymin, ymax, zmin, zmax, 0.3, "green")
    # drawPlane(vectorA, -1 + offset, xmin, xmax, ymin, ymax, zmin, zmax, 0.1, "gray")
    # drawPlane(vectorA, +1 + offset, xmin, xmax, ymin, ymax, zmin, zmax, 0.1, "gray")

    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)
    ax.set_zlim(zmin - 1, zmax + 1)
    
    plt.title(f"a = {a}, b = {b}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend()

    
# DEMO
tempSeed = None

if __name__ == "__main__":
    
    # 2D generation and visualization
    generateParams2D = {
        "objNum": 100,
        "halfSize": 10,
        "featNum": 2,
        "sigma": 1,
        "a": [1, 5],
        "b": 2
    }

    linGenerator = gen.LinearGenerator(tempSeed)
    baseLinearSample = linGenerator.base(
        generateParams2D["objNum"],
        generateParams2D["featNum"],
        generateParams2D["halfSize"],
        generateParams2D["sigma"]        
    )   

    customLinearSample = gen.LinearGenerator(tempSeed).specifiedHyperplane(
        generateParams2D["objNum"],
        generateParams2D["featNum"],
        generateParams2D["halfSize"],
        generateParams2D["sigma"],
        generateParams2D["a"],
        generateParams2D["b"]        
    )

    plot2D(baseLinearSample)
    plot2D(customLinearSample)
    plt.show()

    # # 3D generation and visualization
    # generateParams3D = {
    #     "objNum": 100,
    #     "halfSize": 10,
    #     "featNum": 3,
    #     "sigma": 1,
    #     "a": [1, 0, 0],
    #     "b": -2
    # }

    # linGenerator = gen.LinearGenerator(tempSeed)
    # baseLinearSample = linGenerator.base(
    #     generateParams3D["objNum"],
    #     generateParams3D["featNum"],
    #     generateParams3D["halfSize"],
    #     generateParams3D["sigma"]        
    # )   

    # customLinearSample = gen.LinearGenerator(tempSeed).specifiedHyperplane(
    #     generateParams3D["objNum"],
    #     generateParams3D["featNum"],
    #     generateParams3D["halfSize"],
    #     generateParams3D["sigma"],
    #     generateParams3D["a"],
    #     generateParams3D["b"]        
    # )
    
    # plot3D(baseLinearSample)
    # plot3D(customLinearSample)
    # plt.show()


    # # TXT save and load without generate parametrs
    # generateParams = {
    #     "objNum": 10000,
    #     "halfSize": 100,
    #     "featNum": 100,
    #     "sigma": 1,
    #     "a": None,  # random
    #     "b": -2
    # } 

    # customLinearSample = gen.LinearGenerator(tempSeed).specifiedHyperplane(
    #     generateParams["objNum"],
    #     generateParams["featNum"],
    #     generateParams["halfSize"],
    #     generateParams["sigma"],
    #     generateParams["a"],
    #     generateParams["b"]        
    # ) 

    # customLinearSample.saveTXT(r'D:\customLinearSample.txt')
    
    # sampleFromFile = gen.LinearSample()
    # sampleFromFile.loadTXT(r'D:\customLinearSample.txt')

    # # binary save and load with generate parametrs
    # generateParams = {
    #     "objNum": 11010,
    #     "halfSize": 50,
    #     "featNum": 101,
    #     "sigma": 0.8,
    #     "a": None,  # random
    #     "b": -2.5
    # } 

    # customLinearSample = gen.LinearGenerator(tempSeed).specifiedHyperplane(
    #     generateParams["objNum"],
    #     generateParams["featNum"],
    #     generateParams["halfSize"],
    #     generateParams["sigma"],
    #     generateParams["a"],
    #     generateParams["b"]        
    # )

    # customLinearSample.saveBin(r'D:\customLinearSample.npz')
        
    # sampleFromFile = gen.LinearSample()
    # sampleFromFile.loadBin(r'D:\customLinearSample.npz')    
    
    # print(sampleFromFile.params) # generate parametrs

    pass