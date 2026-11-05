"""MNIST inference benchmarks."""

import argparse
import sys

from utils.benchmark_offline import main as offline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("offline",))
    parser.parse_args(sys.argv[1:2])
    offline(sys.argv[2:])


if __name__ == "__main__":
    main()
