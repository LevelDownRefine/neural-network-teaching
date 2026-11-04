"""Shared argument validation."""

from argparse import ArgumentTypeError


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise ArgumentTypeError("must be a positive integer")
    return number
