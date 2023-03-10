import logging
import matplotlib.pyplot as plt
import pandas as pd
import itertools


logging.basicConfig(level=logging.DEBUG)

FILENAME = "iris.csv"


def scatterplots(irisSetosa, irisVersicolor, irisVirginica, columns):
    for c1, c2 in itertools.combinations(columns[:-1], 2):
        plt.figure()
        plt.scatter(x=irisSetosa[c1], y=irisSetosa[c2], alpha=0.5, label="Iris-setosa")
        plt.scatter(
            x=irisVersicolor[c1],
            y=irisVersicolor[c2],
            alpha=0.5,
            label="Iris-versicolor",
        )
        plt.scatter(
            x=irisVirginica[c1], y=irisVirginica[c2], alpha=0.5, label="Iris-virginica"
        )
        plt.legend()
        plt.xlabel(c1)
        plt.ylabel(c2)
        plt.savefig(f"plots/scatter_{c1}_{c2}.png")
        plt.close()


def histograms(irisSetosa, irisVersicolor, irisVirginica, columns):
    for c in columns[:-1]:
        plt.hist(x=irisSetosa[c], bins=10, alpha=0.5, label="Iris-setosa")
        plt.hist(x=irisVersicolor[c], bins=10, alpha=0.5, label="Iris-versicolor")
        plt.hist(x=irisVirginica[c], bins=10, alpha=0.5, label="Iris-virginica")
        plt.legend()
        plt.xlabel(c)
        plt.savefig(f"plots/hist_{c}.png")
        plt.close()


def statistics(iris):
    for c in iris.columns[:-1]:
        print(f"Statistics for {c}")
        print(iris[c].describe())
        print("")


def visualization():
    iris = pd.read_csv(FILENAME)
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    iris.columns = columns
    plt.figure()
    irisSetosa = iris[iris["species"] == "Iris-setosa"]
    irisVersicolor = iris[iris["species"] == "Iris-versicolor"]
    irisVirginica = iris[iris["species"] == "Iris-virginica"]
    # histograms(irisSetosa, irisVersicolor, irisVirginica, columns)
    # scatterplots(irisSetosa, irisVersicolor, irisVirginica, columns)
    statistics(iris)


if __name__ == "__main__":
    visualization()
