"""MNIST inference benchmarks."""

import argparse
import sys

from utils.benchmark_latency import main as latency
from utils.benchmark_offline import main as offline
from utils.benchmark_power import main as power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("offline", "power", "latency"))
    args = parser.parse_args(sys.argv[1:2])
    {"offline": offline, "power": power, "latency": latency}[args.mode](sys.argv[2:])


if __name__ == "__main__":
    main()
