import logging
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, permutations


logging.basicConfig(level=logging.DEBUG)

FILENAME = "iris.csv"


def mcol(v):
    return v.reshape(v.size, 1)


def load():
    d = []
    c = []
    labels = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    with open(FILENAME, "r") as f:
        for line in f:
            d.append(
                np.array([float(_) for _ in line.strip().split(",")[:-1]]).reshape(4, 1)
            )
            c.append(labels[line.strip().split(",")[-1]])

    dataset = np.hstack(d)
    classes = np.array(c)
    return dataset, classes


def histograms(dataset, classes):
    features = ["sepal length", "sepal width", "petal length", "petal width"]
    for i in range(4):
        plt.figure()
        plt.title(f"Feature {features[i]}")
        plt.hist(
            dataset[i, classes == 0],
            bins=10,
            density=True,
            label="Iris-setosa",
            alpha=0.5,
        )
        plt.hist(
            dataset[i, classes == 1],
            bins=10,
            density=True,
            label="Iris-versicolor",
            alpha=0.5,
        )
        plt.hist(
            dataset[i, classes == 2],
            bins=10,
            density=True,
            label="Iris-virginica",
            alpha=0.5,
        )
        plt.legend()
        plt.savefig(f"plots/NP_hist_{features[i]}.png")
    # plt.show()


def scatterplots(dataset, classes):
    features = ["sepal length", "sepal width", "petal length", "petal width"]
    # for f1, f2 in combinations(range(4), 2):
    for f1, f2 in permutations(range(4), 2):
        plt.figure()
        plt.title(f"Features {features[f1]} vs {features[f2]}")
        plt.scatter(
            dataset[f1, classes == 0],
            dataset[f2, classes == 0],
            label="Iris-setosa",
            alpha=0.5,
        )
        plt.scatter(
            dataset[f1, classes == 1],
            dataset[f2, classes == 1],
            label="Iris-versicolor",
            alpha=0.5,
        )
        plt.scatter(
            dataset[f1, classes == 2],
            dataset[f2, classes == 2],
            label="Iris-virginica",
            alpha=0.5,
        )
        plt.xlabel(features[f1])
        plt.ylabel(features[f2])
        plt.legend()
        plt.savefig(f"plots/NP_scatter_{features[f1]}_{features[f2]}.png")


def visualization():
    dataset, classes = load()
    histograms(dataset, classes)
    scatterplots(dataset, classes)


if __name__ == "__main__":
    visualization()
