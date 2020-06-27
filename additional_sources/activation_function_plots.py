import numpy as np
import matplotlib.pyplot as plt


def logistic_function(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


if __name__ == "__main__":
    X = np.arange(-10.0, 10.0, 0.1)
    Y1 = [logistic_function(i) for i in X]
    Y2 = [np.tanh(i) for i in X]
    Y3 = [relu(i) for i in X]

    plt.title('Функції активації')

    plt.axis([-3, 3, -1, 1])

    plt.plot(X, Y1, linewidth=2.0, label="Sigmoid")
    plt.plot(X, Y2, linewidth=2.0, label="Tanh")
    plt.plot(X, Y3, linewidth=2.0, label="ReLU")

    plt.legend(loc='best', frameon=False)

    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()
    plt.close()
