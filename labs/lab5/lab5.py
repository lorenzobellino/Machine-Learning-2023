import sklearn.datasets
import numpy as np
import argparse
import logging
import scipy

logger = logging.getLogger("lab5")


def vcol(v) -> np.array:
    return v.reshape(v.size, 1)


def vrow(v) -> np.array:
    return v.reshape(1, v.size)


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


def loglikelihood(X, mu, C) -> np.array:
    """Compute the log-likelihood of a set of samples

    Args:
        X (np.Array): (M,N) matrix of N samples of dimension M
        mu (np.Array): array of shape (M,1) representing the mean
        C (np.Array): array of shape (M,M) representig the covariance matrix
    """
    return logpdf_GAU_ND_fast(X, mu, C).sum()


def logpdf_GAU_ND_fast(X, mu, C) -> np.array:
    """Generate the multivariate Gaussian Density for a Matrix of N samples

    Args:
        X (np.Array): (M,N) matrix of N samples of dimension M
        mu (np.Array): array of shape (M,1) representing the mean
        C (np.Array): array of shape (M,M) representig the covariance matrix
    """
    XC = X - mu
    M = X.shape[0]
    invC = np.linalg.inv(C)
    _, logDetC = np.linalg.slogdet(C)
    v = (XC * np.dot(invC, XC)).sum(0)

    lpdf = -(M / 2) * np.log(2 * np.pi) - (1 / 2) * logDetC - (1 / 2) * v
    return lpdf


def MGC(DTR, DTE, LTR, LTE):
    logger.info("dividing the dataset in each class")
    d0 = DTR[:, LTR == 0]
    d1 = DTR[:, LTR == 1]
    d2 = DTR[:, LTR == 2]
    logger.info("computing the mean for each class")
    mean0 = np.mean(d0, axis=1).reshape(-1, 1)
    mean1 = np.mean(d1, axis=1).reshape(-1, 1)
    mean2 = np.mean(d2, axis=1).reshape(-1, 1)
    mu = [mean0, mean1, mean2]
    logger.info("computing the covariance matrix for each class")
    d0c = d0 - mean0
    d1c = d1 - mean1
    d2c = d2 - mean2
    cov0 = np.dot(d0c, d0c.T) / d0.shape[1]
    cov1 = np.dot(d1c, d1c.T) / d1.shape[1]
    cov2 = np.dot(d2c, d2c.T) / d2.shape[1]
    C = [cov0, cov1, cov2]
    logger.debug(f"\nmean0 :\n{mean0}\nmean1 :\n{mean1}\nmean2 :\n{mean2}")
    logger.debug(f"\ncov0 :\n{cov0}\ncov1 :\n{cov1}\ncov2 :\n{cov2}")
    S = []
    for i in range(3):
        ld = logpdf_GAU_ND_fast(DTE, mu[i], C[i])
        ld = [np.exp(x) for x in ld]
        logger.debug(f"Log-density for class {i} : {ld}")
        S.append(ld)
    S = np.array(S)
    # S = scipy.special.logsumexp(S, axis=0)
    logger.debug(f"Log-density matrix :\n{S}")
    logger.info("computing SJoint matrix")
    Sjoint = S * np.array([1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)
    logger.debug(f"SJoint :\n{Sjoint}")
    logger.info("loading a solution")
    Sjoint_MVG = np.load("./solutions/SJoint_MVG.npy")
    logger.debug(f"SJoint_MVG :\n{Sjoint_MVG}")
    if np.allclose(Sjoint, Sjoint_MVG):
        logger.info("Solution is correct")
        logger.debug(f"diff :\n{Sjoint - Sjoint_MVG}")
    else:
        logger.info("Solution is incorrect")
    logger.info("Computing marginal density")
    SMarginal = vrow(Sjoint.sum(0))
    logger.debug(f"SMarginal :\n{SMarginal}")
    logger.info("Computing posterior")
    SPosterior = Sjoint / SMarginal
    logger.debug(f"SPosterior :\n{SPosterior}")
    logger.info("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    logger.info(f"label :\n{label}")
    logger.info("Checking original labels")
    logger.info(f"original label :\n{LTE}")
    logger.info("Compute the accuracy")
    accuracy = np.sum(label == LTE) / LTE.shape[0]
    logger.info(f"accuracy : {accuracy}")
    logger.info("Compute the error rate")
    errorRate = 1 - accuracy
    logger.info(f"error rate : {errorRate}")


def MGC_V2(DTR, DTE, LTR, LTE):
    """Computing the Multivariate Gaussian Classifier

    Args:
        DTR (np.array): Data training set
        DTE (np.array): Data test set
        LTR (np.array): Labels training set
        LTE (np.array): Labels test set
    """
    logger.info("dividing the dataset in each class")
    d0 = DTR[:, LTR == 0]
    d1 = DTR[:, LTR == 1]
    d2 = DTR[:, LTR == 2]
    logger.info("computing the mean for each class")
    mean0 = np.mean(d0, axis=1).reshape(-1, 1)
    mean1 = np.mean(d1, axis=1).reshape(-1, 1)
    mean2 = np.mean(d2, axis=1).reshape(-1, 1)
    mu = [mean0, mean1, mean2]
    logger.info("computing the covariance matrix for each class")
    d0c = d0 - mean0
    d1c = d1 - mean1
    d2c = d2 - mean2
    cov0 = np.dot(d0c, d0c.T) / d0.shape[1]
    cov1 = np.dot(d1c, d1c.T) / d1.shape[1]
    cov2 = np.dot(d2c, d2c.T) / d2.shape[1]
    C = [cov0, cov1, cov2]
    logger.debug(f"\nmean0 :\n{mean0}\nmean1 :\n{mean1}\nmean2 :\n{mean2}")
    logger.debug(f"\ncov0 :\n{cov0}\ncov1 :\n{cov1}\ncov2 :\n{cov2}")
    S = []
    for i in range(3):
        ld = logpdf_GAU_ND_fast(DTE, mu[i], C[i])
        # ld = [np.exp(x) for x in ld]
        logger.debug(f"Log-density for class {i} : {ld}")
        S.append(ld)
    logSjoint = S * np.array([1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)
    # logSjoint = np.array(S)
    logSMarginal = vrow(scipy.special.logsumexp(logSjoint, axis=0))
    logSPosterior = logSjoint - logSMarginal
    SPosterior = np.exp(logSPosterior)
    # logger.debug(f"SPosterior :\n{SPosterior}")
    logger.debug("loading solutions ...")
    logSJoint_MVG = np.load("./solutions/logSJoint_MVG.npy")
    logMarginal_MVG = np.load("./solutions/logMarginal_MVG.npy")
    logSPosterior_MVG = np.load("./solutions/logPosterior_MVG.npy")
    logger.debug(f"logsJoint :n{logSjoint}")
    logger.debug(f"logsJoint_MVG :n{logSJoint_MVG}")
    logger.debug(f"logMarginal :n{logSMarginal}")
    logger.debug(f"logMarginal_MVG :n{logMarginal_MVG}")
    logger.debug(f"logSPosterior :n{logSPosterior}")
    logger.debug(f"logSPosterior_MVG :n{logSPosterior_MVG}")
    if np.allclose(logSjoint, logSJoint_MVG):
        logger.info("Solution for logSJoint is correct")
    else:
        # raise ValueError("Solution for logSJoint is incorrect")
        logger.info("Solution for logSJoint is incorrect")
    if np.allclose(logSMarginal, logMarginal_MVG):
        logger.info("Solution for logMarginal is correct")
    else:
        logger.info("Solution for logMarginal is incorrect")
    if np.allclose(logSPosterior, logSPosterior_MVG):
        logger.info("Solution for logSPosterior is correct")
    else:
        logger.info("Solution for logSPosterior is incorrect")
    logger.info("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    logger.info(f"label :\n{label}")
    logger.info("Checking original labels")
    logger.info(f"original label :\n{LTE}")
    logger.info("Compute the accuracy")
    accuracy = np.sum(label == LTE) / LTE.shape[0]
    logger.info(f"accuracy : {accuracy}")
    logger.info("Compute the error rate")
    errorRate = 1 - accuracy
    logger.info(f"error rate : {errorRate}")


def NBGC(DTR, DTE, LTR, LTE):
    """Computing the Naive Bayes Gaussian Classifier

    Args:
        DTR (np.array): Data training set
        DTE (np.array): Data test set
        LTR (np.array): Labels training set
        LTE (np.array): Labels test set
    """
    logger.info("dividing the dataset in each class")
    d0 = DTR[:, LTR == 0]
    d1 = DTR[:, LTR == 1]
    d2 = DTR[:, LTR == 2]
    logger.info("computing the mean for each class")
    mean0 = np.mean(d0, axis=1).reshape(-1, 1)
    mean1 = np.mean(d1, axis=1).reshape(-1, 1)
    mean2 = np.mean(d2, axis=1).reshape(-1, 1)
    mu = [mean0, mean1, mean2]
    logger.info("computing the covariance matrix for each class")
    d0c = d0 - mean0
    d1c = d1 - mean1
    d2c = d2 - mean2
    cov0 = np.dot(d0c, d0c.T) / d0.shape[1]
    cov1 = np.dot(d1c, d1c.T) / d1.shape[1]
    cov2 = np.dot(d2c, d2c.T) / d2.shape[1]
    logger.info("extracting the diagonal elements of the covariance matrix")
    cov0 = np.diag(np.diag(cov0))
    cov1 = np.diag(np.diag(cov1))
    cov2 = np.diag(np.diag(cov2))
    C = [cov0, cov1, cov2]
    logger.debug(f"\nmean0 :\n{mean0}\nmean1 :\n{mean1}\nmean2 :\n{mean2}")
    logger.debug(f"\ncov0 :\n{cov0}\ncov1 :\n{cov1}\ncov2 :\n{cov2}")
    S = []
    for i in range(3):
        ld = logpdf_GAU_ND_fast(DTE, mu[i], C[i])
        # ld = [np.exp(x) for x in ld]
        logger.debug(f"Log-density for class {i} : {ld}")
        S.append(ld)
    logSjoint = S * np.array([1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)
    # logSjoint = np.array(S)
    logSMarginal = vrow(scipy.special.logsumexp(logSjoint, axis=0))
    logSPosterior = logSjoint - logSMarginal
    SPosterior = np.exp(logSPosterior)
    # logger.debug(f"SPosterior :\n{SPosterior}")
    logger.debug("loading solutions ...")
    logSJoint_MVG = np.load("./solutions/logSJoint_NaiveBayes.npy")
    logMarginal_MVG = np.load("./solutions/logMarginal_NaiveBayes.npy")
    logSPosterior_MVG = np.load("./solutions/logPosterior_NaiveBayes.npy")
    logger.debug(f"logsJoint :n{logSjoint}")
    logger.debug(f"logsJoint_MVG :n{logSJoint_MVG}")
    logger.debug(f"logMarginal :n{logSMarginal}")
    logger.debug(f"logMarginal_MVG :n{logMarginal_MVG}")
    logger.debug(f"logSPosterior :n{logSPosterior}")
    logger.debug(f"logSPosterior_MVG :n{logSPosterior_MVG}")
    if np.allclose(logSjoint, logSJoint_MVG):
        logger.info("Solution for logSJoint is correct")
    else:
        # raise ValueError("Solution for logSJoint is incorrect")
        logger.info("Solution for logSJoint is incorrect")
    if np.allclose(logSMarginal, logMarginal_MVG):
        logger.info("Solution for logMarginal is correct")
    else:
        logger.info("Solution for logMarginal is incorrect")
    if np.allclose(logSPosterior, logSPosterior_MVG):
        logger.info("Solution for logSPosterior is correct")
    else:
        logger.info("Solution for logSPosterior is incorrect")
    logger.info("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    logger.info(f"label :\n{label}")
    logger.info("Checking original labels")
    logger.info(f"original label :\n{LTE}")
    logger.info("Compute the accuracy")
    accuracy = np.sum(label == LTE) / LTE.shape[0]
    logger.info(f"accuracy : {accuracy}")
    logger.info("Compute the error rate")
    errorRate = 1 - accuracy
    logger.info(f"error rate : {errorRate}")


def TCGC(DTR, DTE, LTR, LTE):
    """Computing the Tied Covariance Gaussian Classifier

    Args:
        DTR (np.array): Data training set
        DTE (np.array): Data test set
        LTR (np.array): Labels training set
        LTE (np.array): Labels test set
    """


def main(args) -> None:
    D, L = load_iris()
    DTR, DTE, LTR, LTE = split_db_2to1(D, L)

    if args.type == "MGC":
        logger.info("\n-------------Multivariate Gaussian Classifier---------------")
        MGC(DTR, DTE, LTR, LTE)
        logger.info(
            "\n-------------Multivariate Gaussian Classifier log domain------------"
        )
        MGC_V2(DTR, DTE, LTR, LTE)
    elif args.type == "NBGC":
        logger.info("\n-------------Naive Bayes Gaussian Classifier---------------")
        NBGC(DTR, DTE, LTR, LTE)
    elif args.type == "TCGC":
        logger.info("\n-------------Tied Covariance Gaussian Classifier---------------")
        TCGC(DTR, DTE, LTR, LTE)
        pass
    else:
        logger.error("Invalid type of analysis")
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
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    main(args)
