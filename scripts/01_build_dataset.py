"""CLI entry point: raw futures data -> processed returns + features.

Placeholder script, no implementation yet.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the processed returns + features dataset.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _ = args
    raise NotImplementedError


if __name__ == "__main__":
    main()
