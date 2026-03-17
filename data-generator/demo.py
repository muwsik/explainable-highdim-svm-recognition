import numpy as np
import matplotlib.pyplot as plt

import generator

# 2D visualization
def plot_2d(X, Y):

    pos = X[Y ==  1]
    neg = X[Y == -1]

    plt.figure(figsize = (6, 6))

    plt.scatter(pos[:, 0], pos[:, 1], c = "blue", label = "+1", alpha = 0.6)
    plt.scatter(neg[:, 0], neg[:, 1], c = "red", label = "-1", alpha = 0.6)

    # separating hyperplane
    plt.axvline(0, color = "black", linestyle = "--", label = "x1 = 0")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("2D sample")
    plt.grid(True)


# 3D visualization
def plot_3d(X, Y):

    pos = X[Y ==  1]
    neg = X[Y == -1]

    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c = "blue", label = "+1", alpha = .6)

    ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2],
               c = "red", label = "-1", alpha = 0.6)

    # separating hyperplane
    yy, zz = np.meshgrid(
        np.linspace(-10, 10, 10),
        np.linspace(-10, 10, 10)
    )

    xx = np.zeros_like(yy)

    ax.plot_surface(
        xx, yy, zz,
        alpha = 0.2,
        color = "gray"
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    ax.set_title("3D sample")
    ax.legend()


# DEMO
if __name__ == "__main__":

    N = 500
    a = 10
    c = 0.8

    X2, Y2 = generator.generateLinearSample(N, 2, a, c, seed = 1)
    plot_2d(X2, Y2)

    X3, Y3 = generator.generateLinearSample(N, 3, a, c, seed = 1)
    plot_3d(X3, Y3)

    plt.show()