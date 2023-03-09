import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)


class Competitor:
    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores
        self.finalScore = self.compute_final_score(scores)

    def __eq__(self, other) -> bool:
        return self.finalScore == other.finalScore

    def __lt__(self, other) -> bool:
        return self.finalScore < other.finalScore

    def __str__(self) -> str:
        return f"{self.name} {self.surname} {self.country} -- {self.finalScore:.1f}"

    def compute_final_score(self, scores):
        return sum(scores) - max(scores) - min(scores)


def compute_ranking(filename):
    try:
        assert os.path.exists(filename)
    except AssertionError:
        logging.error("File does not exist - enter a valid filename")
        exit(-1)
    competitors = []
    with open(filename, "r") as f:
        for line in f:
            # print(line)
            name, surname, country = line.split(" ")[:3]
            scores = [float(_) for _ in line.split(" ")[3:]]
            competitor = Competitor(name, surname, country, scores)
            competitors.append(competitor)

    bestScorers = sorted(competitors, reverse=True)
    print("Final ranking:")
    for i, competitor in enumerate(bestScorers[:3]):
        print(f"{i+1}: {competitor}")

    countryScores = dict()
    for c in competitors:
        if c.country in countryScores:
            countryScores[c.country] += c.finalScore
        else:
            countryScores[c.country] = c.finalScore
    bestCountry = max(countryScores)

    print(f"\nBest country:")
    print(f"{bestCountry} Totatl score: {countryScores[bestCountry]:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exercise1 : read a file and compute scores"
    )
    parser.add_argument("filename", help="the file to read")
    args = parser.parse_args()
    compute_ranking(args.filename)
