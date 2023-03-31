import sklearn.datasets
import numpy as np
import argparse
import logging


def load_iris():
    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2 / 3)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return DTR, DTE, LTR, LTE


def MGC(DTR, DTE, LTR, LTE):
    logging.debug("dividing the dataset in each class")
    d0 = DTR[:, LTR == 0]
    d1 = DTR[:, LTR == 1]
    d2 = DTR[:, LTR == 2]
    logging.debug("computing the mean for each class")
    mean0 = np.mean(d0, axis=1)
    mean1 = np.mean(d1, axis=1)
    mean2 = np.mean(d2, axis=1)
    logging.debug("computing the covariance matrix for each class")
    cov0 = np.cov(d0)
    cov1 = np.cov(d1)
    cov2 = np.cov(d2)

    # finisci prima il lab 4


def main(args) -> None:
    D, L = load_iris()
    DTR, DTE, LTR, LTE = split_db_2to1(D, L)

    if args.type == "MGC":
        logging.info("-------------Multivariate Gaussian Classifier---------------")
        MGC(DTR, DTE, LTR, LTE)
    elif args.type == "NBGC":
        pass
    elif args.type == "TCGC":
        pass
    else:
        logging.error("Invalid type of analysis")
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Lab 5 : Generative models for classification",
        description="Choose the type of analysis to perform:\n"
        "\tMGC : Multivariate Gaussian Classifier\n"
        "\tNBGC : Naive Bayes Gaussian Classifier\n"
        "\tTCGC : Tied Covariance Gaussian Classifier\n",
    )
    parser.add_argument(
        "-t", "--type", type=str, help="Type of analysis (MGC, NBGC, TCGC)"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    # parser.add_argument("-m", "--m", type=int, help="Number of eigenvectors to use")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    main(args)
