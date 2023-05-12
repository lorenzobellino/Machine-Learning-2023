import argparse
import logging
import sklearn.datasets
import scipy.optimize as opt
import numpy as np


logger = logging.getLogger("lab7")

def vcol(v):
    return v.reshape((v.size, 1))

def simple_tests():
    var = np.array([0,0])
    f = lambda var : (var[0] +3)**2 + np.sin(var[0]) +(var[1]+ 1)**2
    f_g = lambda var : ((var[0] +3)**2 + np.sin(var[0]) +(var[1]+ 1)**2 , np.array([2*(var[0]+3) + np.cos(var[0]), 2*(var[1]+1)]))
    grad = lambda var : np.array([2*(var[0]+3) + np.cos(var[0]), 2*(var[1]+1)])
    logger.debug("---------------------------------- approx grad ----------------------------------\n")
    results = opt.fmin_l_bfgs_b(func = f, x0 = var, approx_grad=True, iprint=0)
    logger.debug(f"\nminimum: {results[0]}\nobjective value: {results[1]}\nfuncalls: {results[2]['funcalls']}\nd: {results[2]}")
    logger.debug("---------------------------------- separated f & grad ----------------------------------\n")
    results = opt.fmin_l_bfgs_b(func = f, x0 = var, fprime = grad, iprint=0)
    logger.debug(f"\nminimum: {results[0]}\nobjective value: {results[1]}\nfuncalls: {results[2]['funcalls']}\nd: {results[2]}")
    logger.debug("---------------------------------- f & grad in one ----------------------------------\n")
    results = opt.fmin_l_bfgs_b(func = f_g, x0 = var, iprint=0)
    logger.debug(f"\nminimum: {results[0]}\nobjective value: {results[1]}\nfuncalls: {results[2]['funcalls']}\nd: {results[2]}")

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

def load_iris():
    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )
    return D, L

def w_b(v, dim):
    w = vcol(v[0:dim])
    b = v[-1]
    return w,b

def logereg_wrap(DTR, LTR, lamb):
    dim = DTR.shape[0]
    ZTR  = LTR *2.0 - 1.0
    def logreg_obj(v):
        w,b = w_b(v,dim)
        scores = np.dot(w.T, DTR) +b
        loss_per_sample = np.logaddexp(0, -ZTR*scores)
        loss = loss_per_sample.mean() + 0.5 * lamb * np.linalg.norm(w) **2
        # w,b = v[:-1], v[-1]
        # omega = lamb/2 * np.linalg.norm(w)**2

        # n = DTR.shape[1]
        # # J = omega + 1/n * np.sum(np.logaddexp(0, (-2*LTR -1)) * w.T * DTR +b )
        # J = 0
        # # J = np.sum(np.logaddexp(0, (-2*LTR -1) * (w.T * DTR[:,] +b )))
        # for i in range(n):
        #     J += np.logaddexp(0, (-2*LTR[i] -1) * (w.T * DTR[:,i] +b ))
        #     # print(J)
        # # a = lamb/2 * np.linalg.norm(w)**2 + np.sum(np.log(1+np.exp(-LTR*(w.T@DTR+b))))
        # print(J[0])
        # return omega + 1/n * J[0]
        return loss
    return logreg_obj

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

# def accuracy(D, L, w, b):
#     scores = np.dot(w.T, D) + b
#     pred = (scores > 0).astype(int)
#     return (pred == L).mean()



def main(args):
    if args.debug:
        simple_tests()
    
    logger.info("\n---------------------------- Binary Logistic Regression ----------------------------\n")
    logger.info("Loading iris dataset - binary : virginica (label 0) vs versicolor (label 1)")
    # D, L = load_iris()
    D, L = load_iris_binary()
    logger.info("Splitting dataset")
    DTR, DTE, LTR, LTE = split_db_2to1(D, L)
    logger.info("Optimizing fo different values of lambda")
    lambdas = [10e-7 , 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10, 100]
    acc = []
    err_rate = []
    for lamb in lambdas:
        logger.debug("---------------------------------------------------------")
        logreg_obj = logereg_wrap(DTR, LTR, lamb)
        x,f,d = opt.fmin_l_bfgs_b(func = logreg_obj,approx_grad=True, x0 = np.zeros(DTR.shape[0] + 1), iprint=0)
        logger.debug(f"minimum: {x} | objective value: {f:.5f} | funcalls: {d['funcalls']} | d: {d}")
        logger.debug(f"-------------------------SCORES--------------------------\n")
        w,b = w_b(x, DTR.shape[0])
        scores = np.dot(w.T, DTE) + b
        pred = (scores > 0).astype(int)
        acc.append((pred == LTE).mean()*100)
        err_rate.append((pred != LTE).mean()*100)
        logger.debug(f"lambda : {lamb} | accuracy: {(pred == LTE).mean()*100:.2f} % | error rate: {(pred != LTE).mean()*100:.2f}%")
        # input()
    logger.info(f"------------------------------------")
    logger.info(f"|  lambda  | accuracy | error rate  |")
    for l,a,e in zip(lambdas, acc, err_rate):
        logger.info(f"------------------------------------")
        logger.info(f"|  {l:.1e}  |  {a:0.2f} %  |  {e:.2f} %  |")    

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Lab 7 : Logistic regression for models classification",
        description="Perf",
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