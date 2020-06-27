import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    np.random.seed(0)
    n_samples = 30
    degrees = [1, 15]

    true_fun = lambda X: np.cos(1.5 * np.pi * X)
    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.1

    plt.figure(figsize=(9, 4))
    for i in range(len(degrees)):
        ax = plt.subplot(1, len(degrees), i + 1)
        plt.setp(ax, xticks=(), yticks=())

        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                 include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)

        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Модель")
        plt.plot(X_test, true_fun(X_test), label="Істинна функція")
        plt.scatter(X, y, label="Зразки значень")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        if degrees[i] == 1:
            plt.title("Недонавчання")
        if degrees[i] == 15:
            plt.title("Перенавчання")
    plt.show()
