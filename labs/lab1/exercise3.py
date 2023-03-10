import argparse
import os
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO)


class Person:
    def __init__(self, name, surname, birthplace, birthdate) -> None:
        self.name = name
        self.surname = surname
        self.birthplace = birthplace
        self.birthdate = datetime.strptime(birthdate, "%d/%m/%Y")

    def __str__(self) -> str:
        return f"{self.name} {self.surname} {self.birthplace} {self.birthdate}"


def computeData(filename):
    persons = []
    with open(filename, "r") as f:
        for line in f:
            name, surname, place, bdate = (_ for _ in line.strip().split(" "))
            persons.append(Person(name, surname, place, bdate))
    return persons


def compute(filename):
    # months = {1: }
    months = {
        1: "Genuary",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "Semptember",
        10: "October",
        11: "November",
        12: "Dicember",
    }
    persons = computeData(filename)
    birth_per_city = dict()
    birth_per_month = dict()
    for p in persons:
        try:
            birth_per_month[p.birthdate.month] += 1
        except KeyError:
            birth_per_month[p.birthdate.month] = 1
        try:
            birth_per_city[p.birthplace] += 1
        except KeyError:
            birth_per_city[p.birthplace] = 1

    print("Births per city:")
    total_births = 0
    for city, value in birth_per_city.items():
        print(f"{city}: {value}")
        total_births += value
    print("\nBirths per month:")
    for month, value in birth_per_month.items():
        print(f"{months[month]}: {value}")

    average = total_births / len(birth_per_city)
    print(f"\nAverage number of births: {average}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exercise3 : compute distance or average speeds of some bus lines"
    )
    parser.add_argument("filename", help="the file to read")
    args = parser.parse_args()

    try:
        assert os.path.exists(args.filename)
    except AssertionError:
        logging.error("File does not exist - enter a valid filename")
        exit(-1)

    compute(args.filename)
