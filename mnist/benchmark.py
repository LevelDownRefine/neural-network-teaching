"""MNIST inference benchmarks."""

import argparse
import sys

from utils.benchmark_offline import main as offline
from utils.benchmark_power import main as power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("offline", "power"))
    args = parser.parse_args(sys.argv[1:2])
    {"offline": offline, "power": power}[args.mode](sys.argv[2:])


if __name__ == "__main__":
    main()
