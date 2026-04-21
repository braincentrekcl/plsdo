"""CLI entry point for PLS analysis."""

import argparse


def pls_main(argv=None):
    """PLS covariance analysis with statistical testing and visualisation."""
    parser = argparse.ArgumentParser(
        prog="plsdo",
        description="PLS covariance analysis with statistical testing and visualisation.",
    )
    subparsers = parser.add_subparsers(dest="command")
    args = parser.parse_args(argv)
