"""Benchmark offline MNIST throughput and GPU board power."""

from utils.benchmark_common import create_parser, run


def main(argv=None):
    run(create_parser().parse_args(argv))


if __name__ == "__main__":
    main()
