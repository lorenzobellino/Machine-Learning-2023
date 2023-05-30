import argparse
import logging
import sklearn.datasets
import scipy.optimize as opt
import scipy.special as special
import matplotlib.pyplot as plt

import numpy as np


logger = logging.getLogger("lab8c ")

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

def TCGC(DTR, DTE, LTR, LTE, args, logger):
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
    logSMarginal = vrow(special.logsumexp(logSjoint, axis=0))
    logSPosterior = logSjoint - logSMarginal
    SPosterior = np.exp(logSPosterior)
    logger.debug("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    return label

def MGC(DTR, DTE, LTR, LTE, args, logger):
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
    logSMarginal = vrow(special.logsumexp(logSjoint, axis=0))
    logSPosterior = logSjoint - logSMarginal
    SPosterior = np.exp(logSPosterior)
    
    logger.debug("Compute the predicted label of each sample")
    label = np.argmax(SPosterior, axis=0)
    return label

def print_confusion_matrix(logger, confusion_matrix, c) -> None:
    if c == 2:
        logger.info("----------------------------")
        logger.info("         Class             ")
        logger.info("     |  0  |  1  | Err rate")
        logger.info("----------------------------")
        for i in range(2):
            logger.info(f"  {i}  | {int(confusion_matrix[i,0]):3d} | {int(confusion_matrix[i,1]):3d} | {1 - confusion_matrix[i,i]/np.sum(confusion_matrix[i,:]):5.2f}")
    elif c == 3:
        logger.info("---------------------------------")
        logger.info(f"            Class                   ")
        logger.info(f"    |  0  |  1  |  2  | Err rate ")
        logger.info(f"---------------------------------")
        for i in range(3):
            logger.info(f"  {i} | {int(confusion_matrix[i,0]):3d} | {int(confusion_matrix[i,1]):3d} | {int(confusion_matrix[i,2]):3d} | {1 - confusion_matrix[i,i]/np.sum(confusion_matrix[i,:]):5.2f}")
        logger.info(f"---------------------------------") 
        
def basic_confusion_matrix(args,logger) -> None:
    logger.info("###################################################################################")
    logger.info("#------------------------ Performing clculation on -------------------------------#")
    logger.info("#------------------------       Iris Dataset       -------------------------------#")
    logger.info("###################################################################################")
    logger.info("Loading iris dataset ...")
    D, L = load_iris()
    logger.info("Splitting dataset ...")
    DTR, DTE, LTR, LTE = split_db_2to1(D, L)
    logger.info("Computing predictions with MGC ...")
    predictions = MGC(DTR, DTE, LTR, LTE, args, logger)
    logger.info(f"calculate confusion matrix ...")
    confusion_matrix = calculate_confusion_matrix(predictions,LTE,3)
    # confusion_matrix = np.zeros((3, 3))
    # for i in range(3):
    #     for j in range(3):
    #         confusion_matrix[i, j] = np.sum((predictions == i) & (LTE == j))
    print_confusion_matrix(logger, confusion_matrix,3)
    logger.info("Computing predictions with TCGC ...")
    predictions = TCGC(DTR, DTE, LTR, LTE, args, logger)
    logger.info(f"calculate confusion matrix ...")
    confusion_matrix = calculate_confusion_matrix(predictions,LTE,3)
    # confusion_matrix = np.zeros((3, 3))
    # for i in range(3):
    #     for j in range(3):
    #         confusion_matrix[i, j] = np.sum((predictions == i) & (LTE == j))
    print_confusion_matrix(logger, confusion_matrix,3)

    logger.info("###################################################################################")
    logger.info("#------------------------- Performing clculation on ------------------------------#")
    logger.info("#----------------------- Tercets from Divina Commedia ----------------------------#")
    logger.info("###################################################################################")
    logger.info("Loading tercets dataset ...")
    logLikelyhood = np.load("data/commedia_ll.npy")
    labels = np.load("data/commedia_labels.npy")
    logger.info("Calculating the predictions ...")
    predictions = np.argmax(logLikelyhood, axis=0)
    logger.info("Calculating confusion matrix ...")
    confusion_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            confusion_matrix[i, j] = np.sum((predictions == i) & (labels == j))
    print_confusion_matrix(logger,confusion_matrix,3)

def binary_decision(LLR,pi,Cfn,Cfp) -> np.array:
    t = - np.log(pi*Cfn/((1-pi)*Cfp))
    predictions = np.array([1 if llr > t else 0 for llr in LLR])
    return predictions
  
def binary_optimal_bayes(args, logger) -> None:
    logger.info("###################################################################################")
    logger.info("#------------------------- Performing clculation on ------------------------------#")
    logger.info("#--------------------------- Inferno VS Paradiso ---------------------------------#")
    logger.info("###################################################################################")
    logger.info("Loading the Data ...")
    LLR = np.load("data/commedia_llr_infpar.npy")
    L = np.load("data/commedia_labels_infpar.npy")
    pis = [0.5,0.8,0.5,0.8] # posterior probability
    Cfns = [1,1,10,1]       # Cost for False Negative
    Cfps = [1,1,1,10]       # Cost for False Positive
    DCFs = []
    DFCsn = []
    confusions = []
    # for i in range(4):
    for pi,Cfn,Cfp in zip(pis,Cfns,Cfps):
        logger.info(f"Testing for pi = {pi}, Cfn = {Cfn}, Cfp = {Cfp}")
        logger.info("Calculating predictions ... ")
        predictions = binary_decision(LLR,pi,Cfn,Cfp)
        logger.info("Calculating confusion matrix")
        confusion_matrix = calculate_confusion_matrix(predictions,L,2)
        print_confusion_matrix(logger,confusion_matrix,2)
        confusions.append(confusion_matrix)
    
    logger.info("-------------------------")
    logger.info("Evaluating the Models ...")
    logger.info("-------------------------")
    logger.info(f"  pi  | Cfn | Cfp | DCF")
    logger.info("-------------------------")
    for i in range(4):
        logger.debug(f"pi = {pis[i]}, Cfn = {Cfns[i]}, Cfp = {Cfps[i]}")
        logger.debug("Computing False Negative Ratios")
        FNR = confusions[i][0,1]/(confusions[i][0,1]+confusions[i][1,1])
        logger.debug(f"FNR : {FNR}")
        logger.debug("Computing False Positive Ratios")
        FPR = confusions[i][1,0]/(confusions[i][0,0]+confusions[i][1,0])
        logger.debug(f"FPR : {FPR}")
        logger.debug("Computing Detection Cost Function")
        DCF = (pis[i] * Cfns[i] * FNR) + ((1-pis[i])*Cfps[i]*FPR)
        logger.debug(f"DFC : {DCF}")
        DCFs.append(DCF)
        logger.info(f" {pis[i]:.2f} | {Cfns[i]:3d} | {Cfps[i]:3d} | {DCFs[i]:4.3f}")
    logger.info("-------------------------")
    logger.info("Normalizing the DCFs")
    logger.info("-------------------------")
    logger.info(f"  pi  | Cfn | Cfp | DCF")
    logger.info("-------------------------")
    for i in range(4):
        B = min(pis[i]*Cfns[i],(1-pis[i])*Cfps[i])
        DFCsn.append(DCFs[i]/B)
        logger.info(f" {pis[i]:.2f} | {Cfns[i]:3d} | {Cfps[i]:3d} | {DFCsn[i]:4.3f}")
    logger.info("-------------------------")

def binary_decision_MDC(LLR,pi,Cfn,Cfp) -> dict:
    t = - np.log(pi*Cfn/((1-pi)*Cfp))
    predictions = dict()
    for t in LLR:
        predictions[t] = (np.array([1 if llr > t else 0 for llr in LLR]))
    return predictions

def calculate_confusion_matrix(predictions,labels,n=2) -> np.array: 
    cm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            cm[i,j] = np.sum((predictions == i) & (labels == j))
    return cm

def minimum_detection_cost(args, logger) -> None:
    LLR = np.load("data/commedia_llr_infpar.npy")
    L = np.load("data/commedia_labels_infpar.npy")
    pi_ = [0.5,0.8,0.5,0.8] # posterior probability
    Cfn_ = [1,1,10,1]       # Cost for False Negative
    Cfp_ = [1,1,1,10]       # Cost for False Positive
    min_DCFs = []
    for pi,Cfn,Cfp in zip(pi_,Cfn_,Cfp_):
        logger.info(f"Testing for pi = {pi}, Cfn = {Cfn}, Cfp = {Cfp}")
        logger.info("Calculating predictions ... ")
        predictions = binary_decision_MDC(LLR,pi,Cfn,Cfp)
        min_DCF = []
        for t,p_t in predictions.items():
            cm = calculate_confusion_matrix(p_t,L,2)
            FNR = cm[0,1]/(cm[0,1]+cm[1,1])
            FPR = cm[1,0]/(cm[0,0]+cm[1,0])
            DCF = (pi * Cfn * FNR) + ((1-pi)*Cfp*FPR)
            min_DCF.append((DCF,t))
        min_DCF.sort(key=lambda x: x[0])
        min_DCFs.append(min_DCF[0])
        logger.info(f"Minimum DCF : {min_DCF[0][0]}")
        logger.info(f"Threshold : {min_DCF[0][1]}")
    logger.info("-------------------------")
    logger.info("Normalizing the DCFs")
    logger.info("-------------------------")
    logger.info(f"  pi  | Cfn | Cfp | DCF")
    logger.info("-------------------------")
    for i in range(4):
        B = min(pi_[i]*Cfn_[i],(1-pi_[i])*Cfp_[i])
        logger.info(f" {pi_[i]:.2f} | {Cfn_[i]:3d} | {Cfp_[i]:3d} | {min_DCFs[i][0]/B:4.3f}")
    logger.info("-------------------------")
          
def ROC_curves(args, logger) -> None:
    LLR = np.load("data/commedia_llr_infpar.npy")
    L = np.load("data/commedia_labels_infpar.npy")
    pi_ = [0.5,0.8,0.5,0.8] # posterior probability
    Cfn_ = [1,1,10,1]       # Cost for False Negative
    Cfp_ = [1,1,1,10]       # Cost for False Positive
    x = []
    y = []
    for pi,Cfn,Cfp in zip(pi_,Cfn_,Cfp_):
        logger.info(f"Testing for pi = {pi}, Cfn = {Cfn}, Cfp = {Cfp}")
        logger.info("Calculating predictions ... ")
        predictions = binary_decision_MDC(LLR,pi,Cfn,Cfp)
        min_DCF = []
        TPR = []
        FPR = []
        for t,p_t in predictions.items():
            cm = calculate_confusion_matrix(p_t,L,2)
            FNR = cm[0,1]/(cm[0,1]+cm[1,1])
            FPR.append(cm[1,0]/(cm[0,0]+cm[1,0]))
            TPR.append(1-FNR)
        x += FPR
        y += TPR
            # TNR.append(1-FPR)
        plt.figure()
        plt.scatter(FPR,TPR,s=4)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.grid()
        plt.savefig(f"results/{pi}-{Cfn}-{Cfp}_ROC.png")
    plt.figure()
    plt.scatter(FPR,TPR,s=4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.savefig(f"results/ROC_curve.png")
    

    
    
def main(args):
    logger.info("\n\n###################### BASIC CALCULATIONS OF CONFUSION MATRIX ############################\n\n")
    basic_confusion_matrix(args,logger)
    logger.info("\n\n########################## BINARY TASK : OPTIMAL DECISION ################################\n\n")
    binary_optimal_bayes(args,logger)
    logger.info("\n\n####################### BINARY TASK : MINIMUM DETECTION COST #############################\n\n")
    minimum_detection_cost(args,logger)
    logger.info("\n\n############################# BINARY TASK : ROC CURVES ###################################\n\n")
    ROC_curves(args,logger)
    # x = [0,1,2,3]
    # y = [1,2,3,4]
    # plt.plot(x,y)
    # plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Lab 8 : Optimal decisions",
        description="wow",
    )
    # parser.add_argument("-s", "--step", required=True, type=str, help="type of analysis to perform")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-t", "--test", action="store_true", help="Run simple tests")
    args = parser.parse_args()
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    main(args)
