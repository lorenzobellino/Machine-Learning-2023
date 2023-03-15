import logging
import matplotlib.pyplot as plt
import numpy as np
import argparse
from itertools import permutations


logging.basicConfig(level=logging.DEBUG)

FILENAME = "iris.csv"


def mcol(v) -> np.array:
    return v.reshape(v.size, 1)


def mrow(v) -> np.array:
    return v.reshape(1, v.size)


def load() -> None:
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


def histograms(
    dataset: np.array, classes: np.array, root: str = "plots/", save: bool = False
):
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
        if save:
            plt.savefig(f"{root}NP_hist_{features[i]}.png")
        else:
            plt.show()


def scatterplots(
    dataset: np.array, classes: np.array, root: str = "plots/", save: bool = False
):
    features = ["sepal length", "sepal width", "petal length", "petal width"]
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
        if save:
            plt.savefig(f"{root}NP_scatter_{features[f1]}_{features[f2]}.png")
        else:
            plt.show()


def statistics(dataset: np.array, classes: np.array, save: bool) -> None:
    logging.info(f"computing mean : ")
    mu = dataset.mean(axis=1)
    centerd_data = dataset - mu.reshape(dataset.shape[0], 1)
    logging.info("plotting the centered data")
    histograms(dataset=centerd_data, classes=classes, root="plots/centered/", save=save)
    scatterplots(
        dataset=centerd_data, classes=classes, root="plots/centered/", save=save
    )


def visualization(save: bool) -> None:
    d, c = load()
    histograms(dataset=d, classes=c, root="plots/standard/", save=save)
    scatterplots(dataset=d, classes=c, root="plots/standard/", save=save)
    statistics(dataset=d, classes=c, save=save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="lab2 : datset visualization and statistics on Iris Datatset"
    )
    parser.add_argument("-s", "--save", help="save the plots", action="store_true")
    args = parser.parse_args()
    visualization(args.save)
