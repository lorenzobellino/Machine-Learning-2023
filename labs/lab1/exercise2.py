import argparse
import os
import logging


logging.basicConfig(level=logging.INFO)


class Record:
    def __init__(self, busid, lineid, x, y, time) -> None:
        self.busid = busid
        self.lineid = lineid
        self.distance = 0
        self.position = (x, y)
        self.departure = time
        self.arrival = time

    def update(self, x, y, time):
        self.distance += distance(self.position, (x, y))
        self.position = (x, y)
        self.arrival = time

    def __str__(self) -> str:
        return f"busid : {self.busid} - lineid: {self.lineid}\ndistance = {self.distance} : time = {self.arrival-self.departure}"


def computeData(filename):
    records = dict()
    with open(filename, "r") as f:
        for line in f:
            busid, lineid, x, y, time = (int(_) for _ in line.split(" "))
            try:
                records[busid].update(x, y, time)
            except KeyError:
                records[busid] = Record(busid, lineid, x, y, time)

    return records


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def computeBus(filename, busid):
    records = computeData(filename)
    return records[busid].distance


def computeLine(filename, lineid):
    records = computeData(filename)
    distance = 0
    time = 0
    for r in records.values():
        if r.lineid == lineid:
            distance += r.distance
            time += r.arrival - r.departure

    return distance / time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exercise2 : compute distance or average speeds of some bus lines"
    )
    parser.add_argument("filename", help="the file to read")
    parser.add_argument(
        "-b", "--busid", metavar="<busID>", help="the bus ID to compute", type=int
    )
    parser.add_argument(
        "-l", "--lineid", metavar="<lineID>", help="the line ID to compute", type=int
    )
    args = parser.parse_args()

    try:
        assert os.path.exists(args.filename)
    except AssertionError:
        logging.error("File does not exist - enter a valid filename")
        exit(-1)
    try:
        assert (args.busid is not None) ^ (args.lineid is not None)
    except AssertionError:
        logging.error("You must enter a busID OR a lineID")
        exit(-1)

    if args.busid is not None:
        distance = computeBus(args.filename, args.busid)
        print(f"{args.busid} - Total Distance: {distance}")
    elif args.lineid is not None:
        speed = computeLine(args.filename, args.lineid)
        print(f"{args.lineid} - Avg Speed: {speed}")
