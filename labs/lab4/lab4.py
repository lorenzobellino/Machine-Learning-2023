import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger("lab4")


def vcol(v) -> np.array:
    return v.reshape(v.size, 1)


def vrow(v) -> np.array:
    return v.reshape(1, v.size)


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


def logpdf_GAU_ND(X, mu, C) -> np.array:
    """Generate the multivariate Gaussian Density for a Matrix of N samples

    Args:
        X (np.Array): (M,N) matrix of N samples of dimension M
        mu (np.Array): array of shape (M,1) representing the mean
        C (np.Array): array of shape (M,M) representig the covariance matrix
    """
    Y = []
    for i in range(X.shape[1]):
        Y.append(logpdf_GAU_ND_1Sample(X[:, i : i + 1], mu, C))
    return np.array(Y).ravel()


def logpdf_GAU_ND_1Sample(x: np.array, mu: np.array, C: np.array) -> np.array:
    """logarithmic value of the multivariate Gaussian density

    Args:
        x (np.Array): sample
        mu (np.Array): array of shape (M,1) representing the mean
        C (np.Array): array of shape (M,M) representig the covariance matrix
    """
    M = x.shape[0]

    xc = x - mu  # centered sample <-- USE this

    invC = np.linalg.inv(C)  # precision matrix L
    _, logDetC = np.linalg.slogdet(C)  # log determinant of C

    v = np.dot(xc.T, np.dot(invC, xc)).ravel()

    lpdf = -(M / 2) * np.log(2 * np.pi) - (1 / 2) * logDetC - (1 / 2) * v

    # M = x.shape[0]
    # xc = x - mu
    # L = np.linalg.inv(C)
    # logdet = np.linalg.slogdet(C)[1]
    # v = np.dot(xc.T, np.dot(L, xc)).ravel()
    # lpdf = -(M / 2) * np.log(2 * np.pi) - (1 / 2) * logdet - (1 / 2) * v

    return lpdf


def loglikelihood(X, mu, C) -> np.array:
    """Compute the log-likelihood of a set of samples

    Args:
        X (np.Array): (M,N) matrix of N samples of dimension M
        mu (np.Array): array of shape (M,1) representing the mean
        C (np.Array): array of shape (M,M) representig the covariance matrix
    """
    return logpdf_GAU_ND_fast(X, mu, C).sum()


def MGD():
    """Compute the multivariate Gaussian Density"""
    logger.info("Calculating the multivariate Gaussian density and plotting ...")
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1, 1)) * 1.0
    C = np.ones((1, 1)) * 2.0
    logger.debug(f"m = {m}")
    logger.debug(f"C = {C}")
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_fast(vrow(XPlot), m, C)))
    plt.savefig("./plots/llGAU.png")
    logger.info("Comparing solutions ...")
    pdfSol = np.load("solutions/llGAU.npy")
    pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
    logger.info(f"difference in solution1 = {np.abs(pdfSol - pdfGau).max()}")
    XND = np.load("solutions/XND.npy")
    mu = np.load("solutions/muND.npy")
    C = np.load("solutions/CND.npy")
    pdfSol = np.load("solutions/llND.npy")
    pdfGau = logpdf_GAU_ND_fast(XND, mu, C)
    logger.info(f"difference in solution2 = {np.abs(pdfSol - pdfGau).max()}")


def MLE() -> None:
    """Compute the Maximum Likelihood Estimate"""
    logger.info("Calculating the Maximum Likelihood Estimate ...")
    logger.info("Loading samples ...")
    XND = np.load("solutions/XND.npy")
    logger.info("computing mean and covariance ...")
    mu = XND.mean(axis=1).reshape(-1, 1)
    XNDC = XND - mu  # centering the data
    C = np.dot(XNDC, XNDC.T) / XND.shape[1]
    logger.info(f"mu = \n{mu}")
    logger.info(f"C = \n{C}")
    logger.info("Computing the log-likelihood ...")
    ll = loglikelihood(XND, mu, C)
    logger.info(f"Log-likelihood = {ll}")
    logger.info("Loading an example Dataset ...")
    X1D = np.load("solutions/X1D.npy")
    logger.info("computing mean and covariance ...")
    mu = X1D.mean(axis=1).reshape(-1, 1)
    X1DC = X1D - mu  # centering the data
    C = np.dot(X1DC, X1DC.T) / X1D.shape[1]
    logger.info(f"mu = \n{mu}")
    logger.info(f"C = \n{C}")
    logger.info("Computing the log-likelihood ...")
    ll = loglikelihood(X1D, mu, C)
    logger.info(f"Log-likelihood = {ll}")
    logger.info("Plotting the solution ...")
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C)))
    plt.savefig("./plots/ll1D.png")


def main(args) -> None:
    if args.type == "MGD":
        logger.info(
            "------------------Multivariate Gaussian Density---------------------"
        )
        MGD()
    elif args.type == "MLE":
        logger.info(
            "------------------Maximum Likelihood Estimate---------------------"
        )
        MLE()
    else:
        logger.error("Invalid type of analysis")
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Lab 4 :  computing probability densities and ML estimate",
        description="Choose the type of estimates to perform:\n"
        + "\tMGD: Multivariate Gaussian Density\n"
        + "\tMLE: Maximum Likelihood Estimate\n",
    )
    parser.add_argument(
        "-t", "--type", type=str, help="Type of analysis (MGC, NBGC, TCGC)"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main(args)
