import logging
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import argparse


logging.basicConfig(level=logging.DEBUG)

FILENAME = "iris.csv"


def scatterplots(
    irisSetosa: pd.DataFrame,
    irisVersicolor: pd.DataFrame,
    irisVirginica: pd.DataFrame,
    columns: pd.DataFrame,
    root: str,
    save: bool = True,
) -> None:
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
        if save:
            plt.savefig(f"plots/standard_pandas/scatter_{c1}_{c2}.png")
        else:
            plt.show()
        plt.close()


def histograms(
    irisSetosa: pd.DataFrame,
    irisVersicolor: pd.DataFrame,
    irisVirginica: pd.DataFrame,
    columns: pd.DataFrame,
    root: str,
    save: bool = True,
) -> None:
    for c in columns[:-1]:
        plt.hist(x=irisSetosa[c], bins=10, alpha=0.5, label="Iris-setosa")
        plt.hist(x=irisVersicolor[c], bins=10, alpha=0.5, label="Iris-versicolor")
        plt.hist(x=irisVirginica[c], bins=10, alpha=0.5, label="Iris-virginica")
        plt.legend()
        plt.xlabel(c)
        if save:
            plt.savefig(f"{root}hist_{c}.png")
        else:
            plt.show()
        plt.close()


def statistics(iris: pd.DataFrame, save: bool) -> None:
    for c in iris.columns[:-1]:
        print(f"Statistics for {c}")
        print(iris[c].describe())
        print("")


def visualization(save=True) -> None:
    iris = pd.read_csv(FILENAME)
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    iris.columns = columns
    plt.figure()
    irisSetosa = iris[iris["species"] == "Iris-setosa"]
    irisVersicolor = iris[iris["species"] == "Iris-versicolor"]
    irisVirginica = iris[iris["species"] == "Iris-virginica"]
    # histograms(
    #     irisSetosa,
    #     irisVersicolor,
    #     irisVirginica,
    #     columns,
    #     "plots/standard_pandas/",
    #     save,
    # )
    # scatterplots(
    #     irisSetosa,
    #     irisVersicolor,
    #     irisVirginica,
    #     columns,
    #     "plots/standard_pandas/",
    #     save,
    # )
    statistics(iris, save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="lab2 : datset visualization and statistics on Iris Datatset"
    )
    parser.add_argument("-s", "--save", help="save the plots", action="store_true")
    args = parser.parse_args()
    visualization(args.save)
    # visualization()
