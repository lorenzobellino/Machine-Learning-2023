import logging
import argparse
import numpy as np
import itertools
import scipy.special
from collections import namedtuple

logger = logging.getLogger("lab6")


def mcol(v):
    return v.reshape((v.size, 1))


def load_data() -> tuple:
    lInf = []
    with open("data/inferno.txt", encoding="ISO-8859-1") as f:
        for line in f:
            lInf.append(line.strip())
    lPur = []
    with open("data/purgatorio.txt", encoding="ISO-8859-1") as f:
        for line in f:
            lPur.append(line.strip())
    lPar = []
    with open("data/paradiso.txt", encoding="ISO-8859-1") as f:
        for line in f:
            lPar.append(line.strip())

    return lInf, lPur, lPar


def split_data(l, n) -> tuple:
    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])

    return lTrain, lTest


# def word_freq(l, word_dict, eps=0.001) -> dict:
#     # word_count = {}
#     word_count = {word: eps for word in word_dict}
#     # print(word_count)
#     # print(len(word_count))
#     # input()
#     for tercet in l:
#         for word in tercet.split():
#             word_count[word] += 1
#     # for word in word_dict:
#     #     for line in l:
#     #         for w in line.split():
#     #             if w == word:
#     #                 if word in word_count:
#     #                     word_count[word] += 1
#     #                 else:
#     #                     word_count[word] = 1
#     # print(word_count)
#     tot_word = sum(word_count.values())
#     # print(tot_word)
#     # input()
#     word_count = {word: count / tot_word for word, count in word_count.items()}

#     return word_count


def s1_compute_log_frequency(train_set, word_dict, eps=0.001) -> dict:
    word_freq = {}
    for key, train_set in train_set.items():
        word_count = {word: eps for word in word_dict}
        for tercet in train_set:
            for word in tercet.split():
                word_count[word] += 1
        tot_word = sum(word_count.values())
        # word_freq[key] = {word: count / tot_word for word, count in word_count.items()}
        word_freq[key] = {
            word: np.log(count) - np.log(tot_word) for word, count in word_count.items()
        }
    #     print(word_freq[key]["e"])
    #     input()
    # print(word_freq["Inf"]["e"])
    return word_freq


def s1_compute_loglikelyhood(word_log_freq, tercet) -> dict:
    loglikelyhood = {_id: 0 for _id in word_log_freq}
    for _id in word_log_freq:
        for word in tercet.split():
            if word in word_log_freq[_id]:
                loglikelyhood[_id] += word_log_freq[_id][word]
    return loglikelyhood


def s1_compute_log_matrix(word_log_freq, val_set, classes):
    S = np.zeros((len(word_log_freq), len(val_set)))
    for t, tercet in enumerate(val_set):
        h_score = s1_compute_loglikelyhood(word_log_freq, tercet)
        # print(h_score)
        # input()
        for _id, i in classes.items():
            S[i, t] = h_score[_id]
        # print(S)
        # input()
    return S


def compute_classPosteriors(S, logPrior=None):
    """
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    """

    if logPrior is None:
        logPrior = np.log(np.ones(S.shape[0]) / float(S.shape[0]))
    J = S + mcol(logPrior)  # Compute joint probability
    ll = scipy.special.logsumexp(J, axis=0)  # Compute marginal likelihood log f(x)
    P = (
        J - ll
    )  # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return np.exp(P)


def compute_accuracy(P, y):
    PredictedLabel = np.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == y.ravel()).sum()
    NTotal = y.size
    return float(NCorrect) / float(NTotal)


def main(args) -> None:
    # print(args)
    logger.info("\n-----------------------Preprocessing-----------------------\n")
    logger.info("Loading data ... ")
    lInf, lPur, lPar = load_data()
    logger.info("Splitting data ... ")
    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    word_dict = set(
        [x for x in lInf_train + lPur_train + lPar_train for x in x.split()]
    )
    classes = {"Inf": 0, "Pur": 1, "Par": 2}

    l_evaluation = lInf_evaluation + lPur_evaluation + lPar_evaluation

    test_set = {"Inf": lInf_evaluation, "Pur": lPur_evaluation, "Par": lPar_evaluation}

    train_set = {"Inf": lInf_train, "Pur": lPur_train, "Par": lPar_train}
    logger.info(
        "\n-------------------------Starting Computations-------------------------\n"
    )
    logger.info("Computing log frequency ... ")
    word_log_freq = s1_compute_log_frequency(train_set, word_dict, eps=args.eps)
    logger.info("Computing log likelyhood matrix ... ")
    S = s1_compute_log_matrix(word_log_freq, l_evaluation, classes)
    logger.info("Computing class posteriors ... ")
    s1_predictions = compute_classPosteriors(S)
    # print(s1_predictions.shape)
    # print(s1_predictions[classes["Inf"]])
    # input()
    logger.info("Computing labels ... ")
    labelsInf = np.zeros(len(lInf_evaluation))
    labelsInf[:] = classes["Inf"]

    labelsPur = np.zeros(len(lPur_evaluation))
    labelsPur[:] = classes["Pur"]

    labelsPar = np.zeros(len(lPar_evaluation))
    labelsPar[:] = classes["Par"]

    labelsEval = np.hstack([labelsInf, labelsPur, labelsPar])
    # s1_labels = np.argmax(s1_predictions, axis=0)
    logger.info("Computing accuracy ... ")
    for clx in classes:
        logger.info(f"\n--------------- {clx} ---------------")
        accuracy = compute_accuracy(
            s1_predictions[:, labelsEval == classes[clx]],
            labelsEval[labelsEval == classes[clx]],
        )
        logger.info(f"Accuracy: {accuracy*100:.2f} %")

    logger.info("\n------------- Total -------------")
    accuracy = compute_accuracy(s1_predictions, labelsEval)
    logger.info(f"Accuracy: {accuracy*100:.2f} %")

    logger.info(f"\n------------------- Binary Classification -------------------\n")
    for c1, c2 in itertools.combinations(classes.keys(), r=2):
        print(f"\n------------------------- {c1} vs {c2} -------------------------\n")
        bin_classes = {c1: 0, c2: 1}
        bin_train_set = {c1: train_set[c1], c2: train_set[c2]}
        bin_word_dict = set(
            [x for x in bin_train_set[c1] + bin_train_set[c2] for x in x.split()]
        )
        bin_eval = test_set[c1] + test_set[c2]
        bin_word_log_freq = s1_compute_log_frequency(
            bin_train_set, bin_word_dict, eps=args.eps
        )
        bin_S = s1_compute_log_matrix(bin_word_log_freq, bin_eval, bin_classes)
        bin_predictions = compute_classPosteriors(bin_S)

        labelsc1 = np.zeros(len(test_set[c1]))
        labelsc1[:] = bin_classes[c1]

        labelsc2 = np.zeros(len(test_set[c2]))
        labelsc2[:] = bin_classes[c2]

        labelsEval = np.hstack([labelsc1, labelsc2])

        accuracy = compute_accuracy(bin_predictions, labelsEval)
        logger.info(f"Accuracy {c1} VS {c2} : {accuracy*100:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Lab 6 : Generative models for classification of text",
        description="wow",
    )
    parser.add_argument(
        "-l", "--load", action="store_true", help="Load solutions and check correctness"
    )
    parser.add_argument(
        "-e",
        "--eps",
        nargs="?",
        const=0.001,
        type=float,
        help="Epsilon value for smoothing",
        required=True,
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
