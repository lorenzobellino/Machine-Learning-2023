import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

DEBUG = False
FILENAME = "iris.csv"

# logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)


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


def plot(proj: np.array, c: np.array, filename: str) -> None:
    plt.figure()
    plt.scatter(proj[0, c == 0], proj[1, c == 0], label="Iris-setosa")
    plt.scatter(proj[0, c == 1], proj[1, c == 1], label="Iris-versicolor")
    plt.scatter(proj[0, c == 2], proj[1, c == 2], label="Iris-virginica")
    plt.legend()
    # plt.show()
    plt.savefig(filename)


def PCA(m: int = 4) -> None:
    logging.info(
        "\n##########################################################\n#                                                        #\n#                      COMPUTING PCA                     #\n#                                                        #\n##########################################################"
    )
    plotname = f"plots/PCA_matrix_{m}.png"
    logging.info("Loading dataset ... ")
    d, c = load()
    logging.info("Done loading")
    logging.info("Calculating the mean and centering data ... ")
    mean = np.mean(d, axis=1)
    logging.info(f"\n-------------------------mean-----------------------\n{mean}")
    centerd_data = d - mean.reshape((mean.size, 1))
    logging.info("Done centering")
    logging.info("Computing covariance matrix ... ")
    cov = np.dot(centerd_data, centerd_data.T) / d.shape[1]
    logging.info("Done computing covariance matrix")
    logging.info(f"\n-------------------------cov------------------------\n{cov}")
    logging.info("Computing eigenvalues and eigenvectors ... ")
    eigvals, eigvecs = np.linalg.eigh(cov)
    logging.info("Done computing eigenvalues and eigenvectors")
    logging.info(f"\n-----------------------eigvals----------------------\n{eigvals}")
    logging.info(f"\n-----------------------eigvecs----------------------\n{eigvecs}")
    logging.info(f"Retrieving m={m} largest eigenvalues ... ")
    U = eigvecs[:, ::-1]
    P = U[:, 0:m]
    logging.info(f"Done retrieving m={m} largest eigenvalues")
    logging.info(f"\n--------------------------P-------------------------\n{P}")
    logging.info(f"Projecting data onto m={m} eigenvectors ... ")
    proj = np.dot(P.T, d)
    logging.info(f"Done projecting data onto m={m} eigenvectors")
    logging.info("Plotting the data ... ")
    plot(proj, c, plotname)


def PCAv2(m: int) -> None:
    logging.info(
        "\n##########################################################\n#                                                        #\n#                    COMPUTING PCA v2                    #\n#                                                        #\n##########################################################"
    )
    plotname = f"plots/PCA_v2_matrix_{m}.png"
    logging.info("Loading dataset ... ")
    d, c = load()
    logging.info("Done loading")
    logging.info("Calculating the mean and centering data ... ")
    mean = np.mean(d, axis=1)
    logging.info(f"\n-------------------------mean-----------------------\n{mean}")
    centerd_data = d - mean.reshape((mean.size, 1))
    logging.info("Done centering")
    logging.info("Computing covariance matrix ... ")
    cov = np.dot(centerd_data, centerd_data.T) / d.shape[1]
    logging.info(f"\n-------------------------cov------------------------\n{cov}")
    logging.info("Computing eigenvectors ... ")
    eigvecs, _, _ = np.linalg.svd(cov)
    logging.info("Done computing eigenvectors")
    logging.info(f"Retrieving m={m} largest eigenvectors ... ")
    P = eigvecs[:, 0:m]
    logging.info(f"\n--------------------------P-------------------------\n{P}")
    logging.info(f"Done retrieving m={m} largest eigenvectors")
    logging.info(f"Projecting data onto m={m} eigenvectors ... ")
    proj = np.dot(P.T, d)
    logging.info(f"Done projecting data onto m={m} eigenvectors")
    logging.info("Plotting the data ... ")
    plot(proj, c, plotname)


def LDAv2(m: int) -> None:
    plotname = f"plots/LDA_v2_matrix_{m}.png"
    logging.info(
        "\n##########################################################\n#                                                        #\n#                    COMPUTING LDA v2                    #\n#                                                        #\n##########################################################"
    )
    d, c = load()
    logging.info("Separating the dataset in each class")
    d0 = d[:, c == 0]
    d1 = d[:, c == 1]
    d2 = d[:, c == 2]
    nc = [d0.shape[1], d1.shape[1], d2.shape[1]]
    logging.info("Computing the covariance matrix for each class")
    mean = np.mean(d, axis=1)
    mean0 = np.mean(d0, axis=1)
    mean1 = np.mean(d1, axis=1)
    mean2 = np.mean(d2, axis=1)
    class_means = [mean0, mean1, mean2]
    centerd_data0 = d0 - mean0.reshape((mean0.size, 1))
    centerd_data1 = d1 - mean1.reshape((mean1.size, 1))
    centerd_data2 = d2 - mean2.reshape((mean2.size, 1))
    cov0 = np.dot(centerd_data0, centerd_data0.T) / d0.shape[1]
    cov1 = np.dot(centerd_data1, centerd_data1.T) / d1.shape[1]
    cov2 = np.dot(centerd_data2, centerd_data2.T) / d2.shape[1]
    covariances = [cov0, cov1, cov2]
    logging.info("Computing the covariance matrix between class (Sb)")
    Sb = (
        sum(
            c * (m - mean) * (m - mean).reshape((m.size, 1))
            for c, m in zip(nc, class_means)
        )
        / d.shape[1]
    )
    logging.info(f"\n-------------------------Sb------------------------\n{Sb}")
    logging.info("Computing the covariance matrix within class (Sw)")
    Sw = sum(c * cov for c, cov in zip(nc, covariances)) / d.shape[1]
    logging.info(f"\n-------------------------Sw------------------------\n{Sw}")
    logging.info("Solving the generalized eigenvalue problem by joint diagonalization")
    eigvecs, eigval, _ = np.linalg.svd(Sw)

    P1 = np.dot(np.dot(eigvecs, np.diag(1.0 / (eigval**0.5))), eigvecs.T)

    logging.info("Computing the trasformed between class covariance (Sbt)")
    Sbt = P1 * Sb * P1.T
    logging.info(f"\n-------------------------Sbt------------------------\n{Sbt}")
    logging.info("Calculating the eigenvectors of Sbt")
    eigvecs, _, _ = np.linalg.svd(Sbt)
    logging.info(
        f"\n-------------------------eigvecs------------------------\n{eigvecs}"
    )
    logging.info(f"Retrieving m={m} largest eigenvectors ... ")
    # P2 = eigvecs[:, 0:m]
    P2 = eigvecs
    logging.info(f"\n--------------------------P-------------------------\n{P2}")
    logging.info("Calculating the LDA amtrix W")
    W = P1.T * P2
    logging.info(f"\n--------------------------W-------------------------\n{W}")


def LDA(m: int) -> None:
    plotname = f"plots/LDA_matrix_{m}.png"
    logging.info(
        "\n##########################################################\n#                                                        #\n#                      COMPUTING LDA                     #\n#                                                        #\n##########################################################"
    )
    d, c = load()
    logging.info("Separating the dataset in each class")
    d0 = d[:, c == 0]
    d1 = d[:, c == 1]
    d2 = d[:, c == 2]
    nc = [d0.shape[1], d1.shape[1], d2.shape[1]]
    logging.info("Computing the covariance matrix for each class")
    mean = np.mean(d, axis=1)
    mean0 = np.mean(d0, axis=1)
    mean1 = np.mean(d1, axis=1)
    mean2 = np.mean(d2, axis=1)
    class_means = [mean0, mean1, mean2]
    centerd_data0 = d0 - mean0.reshape((mean0.size, 1))
    centerd_data1 = d1 - mean1.reshape((mean1.size, 1))
    centerd_data2 = d2 - mean2.reshape((mean2.size, 1))
    cov0 = np.dot(centerd_data0, centerd_data0.T) / d0.shape[1]
    cov1 = np.dot(centerd_data1, centerd_data1.T) / d1.shape[1]
    cov2 = np.dot(centerd_data2, centerd_data2.T) / d2.shape[1]
    covariances = [cov0, cov1, cov2]
    logging.info("Computing the covariance matrix between class (Sb)")
    Sb = (
        sum(
            c * (m - mean) * (m - mean).reshape((m.size, 1))
            for c, m in zip(nc, class_means)
        )
        / d.shape[1]
    )
    logging.info(f"\n-------------------------Sb------------------------\n{Sb}")
    logging.info("Computing the covariance matrix within class (Sw)")
    Sw = sum(c * cov for c, cov in zip(nc, covariances)) / d.shape[1]
    logging.info(f"\n-------------------------Sw------------------------\n{Sw}")
    logging.info("Solving the generalized eigenvalue problem")
    # cannot use np.linalg.eigh
    # because it does not support generalized eigenvalue problem
    eigvals, eigvecs = linalg.eigh(Sb, Sw)
    U = eigvecs[:, ::-1]
    logging.info(f"Retrieving {m} largest eigenvectors")
    W = U[:, 0:m]
    logging.info(f"\n-------------------------W------------------------\n{W}")
    logging.info(f"Projecting data onto {m} eigenvectors")
    proj = np.dot(W.T, d)
    logging.info("Plotting the data")
    plot(proj, c, plotname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 3 : Principal Component Analysis")
    parser.add_argument("-t", "--type", type=str, help="Type of analysis (PCA or LDA)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-m", "--m", type=int, help="Number of eigenvectors to use")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if args.type == "PCA":
        PCA(m=args.m)
        PCAv2(m=args.m)
    elif args.type == "LDA":
        LDA(m=2)
        LDAv2(m=2)
    else:
        logging.error("Invalid type of analysis")
