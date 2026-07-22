"""rl.cli — Command-Line Interface (P4-008 modular wrapper).

P4-008 ROOT FIX: modular wrapper around the CLI entry point (main)
and the argument parser (_build_arg_parser). See rl/env.py for the
full P4-008 rationale.

Callers can now import:
    from rl.cli import main, _build_arg_parser

Or run directly:
    python -m rl.cli --timesteps 50000 --tenant rare_partner
"""
from __future__ import annotations

from .rl_drug_ranker import main, _build_arg_parser

__all__ = ["main", "_build_arg_parser"]


if __name__ == "__main__":
    import sys
    sys.exit(main())
