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


def split_db_loo(D, L, i):
    DTR = np.delete(D, i, axis=1)
    DTE = vcol(D[:, i])
    LTR = np.delete(L, i)
    LTE = L[i]
    return DTR, DTE, LTR, LTE


def split_db_kfold(D, L, k):
    # divide D and L in k folds
    n = D.shape[1]
    nFold = n // k
    DTR = []
    DTE = []
    LTR = []
    LTE = []
    for i in range(k):
        idx = np.arange(i * nFold, (i + 1) * nFold)
        DTR.append(np.delete(D, idx, axis=1))
        DTE.append(vcol(D[:, idx]))
        LTR.append(np.delete(L, idx))
        LTE.append(L[idx])
    idx = np.arange(k * nFold, n)
    if len(idx) > 0:
        DTR.append(np.delete(D, idx, axis=1))
        DTE.append(vcol(D[:, idx]))
        LTR.append(np.delete(L, idx))
        LTE.append(L[idx])
    return DTR, DTE, LTR, LTE


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


def MGC(DTR, DTE, LTR, LTE, args):
    logger.debug("dividing the dataset in each class")
    d0 = DTR[:, LTR == 0]
    d1 = DTR[:, LTR == 1]
    d2 = DTR[:, LTR == 2]
    logger.debug("computing the mean for each class")
    mean0 = np.mean(d0, axis=1).reshape(-1, 1)
    mean1 = np.mean(d1, axis=1).reshape(-1, 1)
    mean2 = np.mean(d2, axis=1).reshape(-1, 1)
    mu = [mean0, mean1, mean2]
    logger.debug("computing the covariance matrix for each class")
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
    logger.debug("computing SJoint matrix")
    Sjoint = S * np.array([1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)
    logger.debug(f"SJoint :\n{Sjoint}")
    logger.debug("loading a solution")
    if args.load:
        Sjoint_MVG = np.load("./solutions/SJoint_MVG.npy")
        logger.debug(f"SJoint_MVG :\n{Sjoint_MVG}")
        if np.allclose(Sjoint, Sjoint_MVG):
            logger.debug("Solution is correct")
            logger.debug(f"diff :\n{Sjoint - Sjoint_MVG}")
        else:
            logger.debug("Solution is incorrect")
    logger.debug("Computing marginal density")
    SMarginal = vrow(Sjoint.sum(0))
    logger.debug(f"SMarginal :\n{SMarginal}")
    logger.debug("Computing posterior")
    SPosterior = Sjoint / SMarginal
    logger.debug(f"SPosterior :\n{SPosterior}")
    logger.debug("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    logger.debug(f"label :\n{label}")
    logger.debug("Checking original labels")
    logger.debug(f"original label :\n{LTE}")
    logger.debug("Compute the accuracy")
    try:
        accuracy = np.sum(label == LTE) / LTE.shape[0]
        errorRate = 1 - accuracy
        if args.type != "KFCV":
            logger.info(f"accuracy : {accuracy:.2f}")
            logger.info(f"error rate : {errorRate:.2f}")
    except IndexError:
        accuracy = np.sum(label == LTE)
    logger.debug("Compute the error rate")
    return accuracy


def MGC_V2(DTR, DTE, LTR, LTE, args):
    """Computing the Multivariate Gaussian Classifier

    Args:
        DTR (np.array): Data training set
        DTE (np.array): Data test set
        LTR (np.array): Labels training set
        LTE (np.array): Labels test set
    """
    logger.debug("dividing the dataset in each class")
    d0 = DTR[:, LTR == 0]
    d1 = DTR[:, LTR == 1]
    d2 = DTR[:, LTR == 2]
    logger.debug("computing the mean for each class")
    mean0 = np.mean(d0, axis=1).reshape(-1, 1)
    mean1 = np.mean(d1, axis=1).reshape(-1, 1)
    mean2 = np.mean(d2, axis=1).reshape(-1, 1)
    mu = [mean0, mean1, mean2]
    logger.debug("computing the covariance matrix for each class")
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
        logger.debug(f"Log-density for class {i} : {ld}")
        S.append(np.array(ld))
    logSjoint = np.array(S) + vcol(
        np.array([np.log(1 / 3), np.log(1 / 3), np.log(1 / 3)])
    )
    logSMarginal = vrow(scipy.special.logsumexp(logSjoint, axis=0))
    logSPosterior = logSjoint - logSMarginal
    SPosterior = np.exp(logSPosterior)
    if args.load:
        logger.debug("loading solutions ...")
        logSJoint_MVG = np.load("./solutions/logSJoint_MVG.npy")
        logMarginal_MVG = np.load("./solutions/logMarginal_MVG.npy")
        logSPosterior_MVG = np.load("./solutions/logPosterior_MVG.npy")

        if np.allclose(logSjoint, logSJoint_MVG):
            logger.debug("Solution for logSJoint is correct")
        else:
            logger.debug(f"logsJoint :n{logSjoint}")
            logger.debug(f"logsJoint_MVG :n{logSJoint_MVG}")
            logger.debug("Solution for logSJoint is incorrect")
            raise ValueError("Solution for logSJoint is incorrect")
        if np.allclose(logSMarginal, logMarginal_MVG):
            logger.debug("Solution for logMarginal is correct")
        else:
            logger.debug(f"logMarginal :n{logSMarginal}")
            logger.debug(f"logMarginal_MVG :n{logMarginal_MVG}")
            logger.debug("Solution for logMarginal is incorrect")
            raise ValueError("Solution for logMarginal is incorrect")
        if np.allclose(logSPosterior, logSPosterior_MVG):
            logger.debug("Solution for logSPosterior is correct")
        else:
            logger.debug(f"logSPosterior :n{logSPosterior}")
            logger.debug(f"logSPosterior_MVG :n{logSPosterior_MVG}")
            logger.debug("Solution for logSPosterior is incorrect")
            raise ValueError("Solution for logSPosterior is incorrect")
    logger.debug("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    logger.debug(f"label :\n{label}")
    logger.debug("Checking original labels")
    logger.debug(f"original label :\n{LTE}")
    logger.debug("Compute the accuracy")
    try:
        accuracy = np.sum(label == LTE) / LTE.shape[0]
        errorRate = 1 - accuracy
        if args.type != "KFCV":
            logger.info(f"accuracy : {accuracy:.2f}")
            logger.info(f"error rate : {errorRate:.2f}")
    except IndexError:
        accuracy = np.sum(label == LTE)
    logger.debug("Compute the error rate")
    return accuracy


def NBGC(DTR, DTE, LTR, LTE, args):
    """Computing the Naive Bayes Gaussian Classifier

    Args:
        DTR (np.array): Data training set
        DTE (np.array): Data test set
        LTR (np.array): Labels training set
        LTE (np.array): Labels test set
    """
    logger.debug("dividing the dataset in each class")
    d0 = DTR[:, LTR == 0]
    d1 = DTR[:, LTR == 1]
    d2 = DTR[:, LTR == 2]
    logger.debug("computing the mean for each class")
    mean0 = np.mean(d0, axis=1).reshape(-1, 1)
    mean1 = np.mean(d1, axis=1).reshape(-1, 1)
    mean2 = np.mean(d2, axis=1).reshape(-1, 1)
    mu = [mean0, mean1, mean2]
    logger.debug("computing the covariance matrix for each class")
    d0c = d0 - mean0
    d1c = d1 - mean1
    d2c = d2 - mean2
    cov0 = np.dot(d0c, d0c.T) / d0.shape[1]
    cov1 = np.dot(d1c, d1c.T) / d1.shape[1]
    cov2 = np.dot(d2c, d2c.T) / d2.shape[1]
    logger.debug("extracting the diagonal elements of the covariance matrix")
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
    logSjoint = np.array(S) + vcol(
        np.array([np.log(1 / 3), np.log(1 / 3), np.log(1 / 3)])
    )
    logSMarginal = vrow(scipy.special.logsumexp(logSjoint, axis=0))
    logSPosterior = logSjoint - logSMarginal
    SPosterior = np.exp(logSPosterior)
    # logger.debug(f"SPosterior :\n{SPosterior}")
    if args.load:
        logger.debug("loading solutions ...")
        logSJoint_MVG = np.load("./solutions/logSJoint_NaiveBayes.npy")
        logMarginal_MVG = np.load("./solutions/logMarginal_NaiveBayes.npy")
        logSPosterior_MVG = np.load("./solutions/logPosterior_NaiveBayes.npy")
        if np.allclose(logSjoint, logSJoint_MVG):
            logger.debug("Solution for logSJoint is correct")
        else:
            logger.debug("Solution for logSJoint is incorrect")
            logger.debug(f"logsJoint :n{logSjoint}")
            logger.debug(f"logsJoint_MVG :n{logSJoint_MVG}")
            raise ValueError("Solution for logSJoint is incorrect")
        if np.allclose(logSMarginal, logMarginal_MVG):
            logger.debug("Solution for logMarginal is correct")
        else:
            logger.debug(f"logMarginal :n{logSMarginal}")
            logger.debug(f"logMarginal_MVG :n{logMarginal_MVG}")
            logger.debug("Solution for logMarginal is incorrect")
            raise ValueError("Solution for logMarginal is incorrect")
        if np.allclose(logSPosterior, logSPosterior_MVG):
            logger.debug("Solution for logSPosterior is correct")
        else:
            logger.debug(f"logSPosterior :n{logSPosterior}")
            logger.debug(f"logSPosterior_MVG :n{logSPosterior_MVG}")
            logger.debug("Solution for logSPosterior is incorrect")
            logger.debug(f"{logSjoint-logSJoint_MVG}")
            raise ValueError("Solution for logSPosterior is incorrect")
    logger.debug("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    logger.debug(f"label :\n{label}")
    logger.debug("Checking original labels")
    logger.debug(f"original label :\n{LTE}")
    logger.debug("Compute the accuracy")
    try:
        accuracy = np.sum(label == LTE) / LTE.shape[0]
        errorRate = 1 - accuracy
        if args.type != "KFCV":
            logger.info(f"accuracy : {accuracy:.2f}")
            logger.info(f"error rate : {errorRate:.2f}")
    except IndexError:
        accuracy = np.sum(label == LTE)
    logger.debug("Compute the error rate")
    return accuracy


def TCGC(DTR, DTE, LTR, LTE, args):
    """Computing the Tied Covariance Gaussian Classifier

    Args:
        DTR (np.array): Data training set
        DTE (np.array): Data test set
        LTR (np.array): Labels training set
        LTE (np.array): Labels test set
    """
    # computing the within-class covariance matrix
    logger.debug("dividing the dataset in each class")
    d0 = DTR[:, LTR == 0]
    d1 = DTR[:, LTR == 1]
    d2 = DTR[:, LTR == 2]
    nc = [d0.shape[1], d1.shape[1], d2.shape[1]]
    logger.debug("computing the mean for each class")
    mean = np.mean(DTR, axis=1).reshape(-1, 1)
    mean0 = np.mean(d0, axis=1).reshape(-1, 1)
    mean1 = np.mean(d1, axis=1).reshape(-1, 1)
    mean2 = np.mean(d2, axis=1).reshape(-1, 1)
    mu = [mean0, mean1, mean2]
    logger.debug("computing the covariance matrix for each class")
    d0c = d0 - mean0
    d1c = d1 - mean1
    d2c = d2 - mean2
    cov0 = np.dot(d0c, d0c.T) / d0.shape[1]
    cov1 = np.dot(d1c, d1c.T) / d1.shape[1]
    cov2 = np.dot(d2c, d2c.T) / d2.shape[1]
    C = [cov0, cov1, cov2]
    logger.debug("Computing the covariance matrix within classes")
    Sw = sum(c * cov for c, cov in zip(nc, C)) / DTR.shape[1]
    Sw_v2 = (1 / DTR.shape[1]) * (
        (LTR == 0).sum() * cov0 + (LTR == 1).sum() * cov1 + (LTR == 2).sum() * cov2
    )
    S = []
    for i in range(3):
        ld = logpdf_GAU_ND_fast(DTE, mu[i], Sw)
        logger.debug(f"Log-density for class {i} : {ld}")
        S.append(np.array(ld))
    logSjoint = np.array(S) + vcol(
        np.array([np.log(1 / 3), np.log(1 / 3), np.log(1 / 3)])
    )
    logSMarginal = vrow(scipy.special.logsumexp(logSjoint, axis=0))
    logSPosterior = logSjoint - logSMarginal
    SPosterior = np.exp(logSPosterior)
    if args.load:
        logger.debug("loading solutions ...")
        logSJoint_MVG = np.load("./solutions/logSJoint_TiedMVG.npy")
        logMarginal_MVG = np.load("./solutions/logMarginal_TiedMVG.npy")
        logSPosterior_MVG = np.load("./solutions/logPosterior_TiedMVG.npy")
        if np.allclose(logSjoint, logSJoint_MVG):
            logger.debug("Solution for logSJoint is correct")
        else:
            logger.debug(f"logsJoint :n{logSjoint}")
            logger.debug(f"logsJoint_MVG :n{logSJoint_MVG}")
            logger.debug("Solution for logSJoint is incorrect")
            raise ValueError("Solution for logSJoint is incorrect")
        if np.allclose(logSMarginal, logMarginal_MVG):
            logger.debug("Solution for logMarginal is correct")
        else:
            logger.debug(f"logMarginal :n{logSMarginal}")
            logger.debug(f"logMarginal_MVG :n{logMarginal_MVG}")
            logger.debug("Solution for logMarginal is incorrect")
            raise ValueError("Solution for logMarginal is incorrect")
        if np.allclose(logSPosterior, logSPosterior_MVG):
            logger.debug("Solution for logSPosterior is correct")
        else:
            logger.debug(f"logSPosterior :n{logSPosterior}")
            logger.debug(f"logSPosterior_MVG :n{logSPosterior_MVG}")
            logger.debug("Solution for logSPosterior is incorrect")
            raise ValueError("Solution for logSPosterior is incorrect")
    logger.debug("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    logger.debug(f"label :\n{label}")
    logger.debug("Checking original labels")
    logger.debug(f"original label :\n{LTE}")
    logger.debug("Compute the accuracy")
    try:
        accuracy = np.sum(label == LTE) / LTE.shape[0]
        errorRate = 1 - accuracy
        if args.type != "KFCV":
            logger.info(f"accuracy : {accuracy:.2f}")
            logger.info(f"error rate : {errorRate:.2f}")
    except IndexError:
        accuracy = np.sum(label == LTE)
    logger.debug("Compute the error rate")
    return accuracy


# def KFCV_LOO(D, L, args):
#     """Compute K-Fold Cross Validation usign a Leave One Out (LOO) approach

#     Args:
#         D (np.array): Dataset
#         L (np.array): Labels
#     """
#     # TODO: Implement the K-Fold Cross Validation
#     accurate_prediction_MGC = 0
#     accurate_prediction_NBGC = 0
#     accurate_prediction_TCGC = 0
#     for i in range(D.shape[1]):
#         logger.debug(f"Fold {i+1}")
#         DTR, DTE, LTR, LTE = split_db_loo(D, L, i)
#         # print("------------------------------------------------")
#         # print(DTR)
#         # print(DTE)
#         # print(LTR)
#         # print(LTE)
#         # input()
#         accurate_prediction_MGC += MGC(DTR, DTE, LTR, LTE, args)
#         accurate_prediction_NBGC += NBGC(DTR, DTE, LTR, LTE, args)
#         accurate_prediction_TCGC += TCGC(DTR, DTE, LTR, LTE, args)
#     accuracy_MGC = accurate_prediction_MGC / L.size * 100
#     accuracy_NBGC = accurate_prediction_NBGC / L.size * 100
#     accuracy_TCGC = accurate_prediction_TCGC / L.size * 100
#     logger.info(f"Accuracy Multivariate Gaussian Classifier: {accuracy_MGC:.2f}%")
#     logger.info(f"Accuracy Naive Bayes Gaussian Classifier: {accuracy_NBGC:.2f}%")
#     logger.info(f"Accuracy Tied Covariance Gaussian Classifier: {accuracy_TCGC:.2f}%")
#     error_rate_MGC = 100 - accuracy_MGC
#     error_rate_NBGC = 100 - accuracy_NBGC
#     error_rate_TCGC = 100 - accuracy_TCGC
#     logger.info(f"Error rate MGC: {error_rate_MGC:.2f}%")
#     logger.info(f"Error rate NBGC: {error_rate_NBGC:.2f}%")
#     logger.info(f"Error rate TCGC: {error_rate_TCGC:.2f}%")
#     logger.info(
#         f"Best classifier: {np.argmin([error_rate_MGC, error_rate_NBGC, error_rate_TCGC])}"
#     )


def KFCV(D, L, args, k):
    """Compute K-Fold Cross Validation usign a Leave One Out (LOO) approach

    Args:
        D (np.array): Dataset
        L (np.array): Labels
    """
    # TODO: Implement the K-Fold Cross Validation
    accurate_prediction_MGC = 0
    accurate_prediction_NBGC = 0
    accurate_prediction_TCGC = 0
    DTR_l, DTE_l, LTR_l, LTE_l = split_db_kfold(D, L, k)
    for DTR, DTE, LTR, LTE in zip(DTR_l, DTE_l, LTR_l, LTE_l):
        # print(DTR)
        # print(DTE)
        # print(LTR)
        # print(LTE)
        # input()
        accurate_prediction_MGC += MGC(DTR, DTE, LTR, LTE, args)
        accurate_prediction_NBGC += NBGC(DTR, DTE, LTR, LTE, args)
        accurate_prediction_TCGC += TCGC(DTR, DTE, LTR, LTE, args)

    # for i in range(D.shape[1]):
    #     logger.debug(f"Fold {i+1}")
    #     DTR, DTE, LTR, LTE = split_db_loo(D, L, i)
    #     accurate_prediction_MGC += MGC(DTR, DTE, LTR, LTE, args)
    #     accurate_prediction_NBGC += NBGC(DTR, DTE, LTR, LTE, args)
    #     accurate_prediction_TCGC += TCGC(DTR, DTE, LTR, LTE, args)
    accuracy_MGC = accurate_prediction_MGC / L.size * 100
    accuracy_NBGC = accurate_prediction_NBGC / L.size * 100
    accuracy_TCGC = accurate_prediction_TCGC / L.size * 100
    logger.info(f"Accuracy Multivariate Gaussian Classifier: {accuracy_MGC:.2f}%")
    logger.info(f"Accuracy Naive Bayes Gaussian Classifier: {accuracy_NBGC:.2f}%")
    logger.info(f"Accuracy Tied Covariance Gaussian Classifier: {accuracy_TCGC:.2f}%")
    error_rate_MGC = 100 - accuracy_MGC
    error_rate_NBGC = 100 - accuracy_NBGC
    error_rate_TCGC = 100 - accuracy_TCGC
    logger.info(f"Error rate MGC: {error_rate_MGC:.2f}%")
    logger.info(f"Error rate NBGC: {error_rate_NBGC:.2f}%")
    logger.info(f"Error rate TCGC: {error_rate_TCGC:.2f}%")
    logger.info(
        f"Best classifier: {np.argmin([error_rate_MGC, error_rate_NBGC, error_rate_TCGC])}"
    )


def main(args) -> None:
    D, L = load_iris()
    DTR, DTE, LTR, LTE = split_db_2to1(D, L)

    if args.type == "MGC":
        logger.info("\n-------------Multivariate Gaussian Classifier---------------")
        MGC(DTR, DTE, LTR, LTE, args)
        logger.info(
            "\n-------------Multivariate Gaussian Classifier log domain------------"
        )
        MGC_V2(DTR, DTE, LTR, LTE, args)
    elif args.type == "NBGC":
        logger.info("\n-------------Naive Bayes Gaussian Classifier---------------")
        NBGC(DTR, DTE, LTR, LTE, args)
    elif args.type == "TCGC":
        logger.info("\n-------------Tied Covariance Gaussian Classifier---------------")
        TCGC(DTR, DTE, LTR, LTE, args)
    elif args.type == "KFCV":
        logger.info("\n-------------K-Fold Cross Validation---------------")
        # KFCV(D, L, args, D.shape[1])
        KFCV(D, L, args, 5)
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
        "\tTCGC : Tied Covariance Gaussian Classifier\n"
        "\tKFCV : K-Fold Cross Validation\n",
    )
    parser.add_argument(
        "-t", "--type", type=str, help="Type of analysis (MGC, NBGC, TCGC, KFCV)"
    )
    parser.add_argument(
        "-l", "--load", action="store_true", help="Load solutions and check correctness"
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
