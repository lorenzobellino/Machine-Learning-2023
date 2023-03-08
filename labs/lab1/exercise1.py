import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)


class Competitor:
    def __init__(self, name, surname, country, scores):
        try:
            assert len(scores) == 5
        except AssertionError:
            logging.error("Scores must have 5 elements -- skipping entry")
            return None
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores


def compute_ranking(filename):
    try:
        assert os.path.exists(filename)
    except AssertionError:
        logging.error("File does not exist - enter a valid filename")
        exit(-1)
    competitors = []
    with open(filename, "r") as f:
        for line in f:
            print(line)
            name, surname, country = line.split(" ")[:3]
            scores = [float(_) for _ in line.split(" ")[3:]]
            competitor = Competitor(name, surname, country, scores)
            if competitor is not None:
                competitors.append(competitor)
    # for competitor in competitors:
    #     print(
    #         competitor.name, competitor.surname, competitor.country, competitor.scores
    #     )
    for c in competitors:
        print(f"name : {c.name}\nsurname : {c.surname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exercise1 : read a file and compute scores"
    )
    parser.add_argument("filename", help="the file to read")
    args = parser.parse_args()
    compute_ranking(args.filename)
