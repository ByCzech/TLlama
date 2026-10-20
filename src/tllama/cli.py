"""Client entry point; importing it never constructs a server."""

from __future__ import annotations

import argparse
from typing import Sequence

from tllama.version import get_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tllama", description="TLlama HTTP client.")
    parser.add_argument("--version", action="version", version=f"tllama {get_version()}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
