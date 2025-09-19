"""Module entry point for ``python -m hairf``."""

from __future__ import annotations

from .cli import run_cli


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
