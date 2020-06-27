import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

if __name__ == "__main__":
    # line
    points = [(1.0, 10.0), (3.0, 2.0), (5.0, 6.0), (7.0, 5.0), (15.0, 12.0)]
    data = np.array(points)
    tck, u = interpolate.splprep(data.transpose(), s=0)
    unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)

    # descend points
    path1_X = [1.1, 3.9, 1.7, 3.0, 2.1, 2.6, 2.3]
    path1_Y = [9, 5, 3.7, 2, 0.5, 0.1, -0.4]

    path2_X = [6, 8, 9.5, 10.9, 12, 12.6, 12.9]
    path2_Y = [5.7, 4.1, 2.6, 1.5, 0.85, 0.7, 0.7]

    # render
    plt.title('Метод стохастичного градієнтного спуску')

    plt.plot(out[0], out[1], color='gray')

    plt.plot(path1_X, path1_Y, color='red', linewidth=1.0)
    for (x, y) in zip(path1_X, path1_Y):
        plt.plot(x, y, 'ob', color='orange')

    plt.plot(path2_X, path2_Y, color='red', linewidth=1.0)
    for (x, y) in zip(path2_X, path2_Y):
        plt.plot(x, y, 'ob', color='blue')

    plt.show()
