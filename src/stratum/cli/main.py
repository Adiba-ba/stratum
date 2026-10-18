"""Command-line entry point for STRATUM.

Subcommands are registered here and implemented in sibling modules so the
package can be used without invoking any script file directly.
"""

from __future__ import annotations

import argparse
import importlib
import sys

__all__ = ["main"]

#: Subcommand name -> (module, callable). Modules are imported lazily so
#: that optional dependencies of one command never block the others.
_COMMANDS: dict[str, tuple[str, str]] = {
    "verify-baseline": ("stratum.cli.verify", "verify_baseline"),
    "screen-routes": ("stratum.cli.verify", "screen_routes"),
    "corridor": ("stratum.cli.corridor", "corridor"),
    "profile": ("stratum.cli.profile", "profile"),
    "build-physics-labels": ("stratum.cli.labels", "build_physics_labels"),
    "build-observed-labels": ("stratum.cli.labels", "build_observed_labels"),
    "train": ("stratum.cli.train", "train"),
    "detect-scene": ("stratum.cli.scene", "detect_scene"),
    "fetch": ("stratum.cli.fetch", "fetch"),
    "check-live": ("stratum.cli.live", "check_live"),
    "demo": ("stratum.cli.demos", "demo"),
}


def main(argv: list[str] | None = None) -> int:
    """Run the STRATUM command line interface."""
    parser = argparse.ArgumentParser(
        prog="stratum",
        description="Contrail persistence prediction and altitude-dispatch evaluation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name in _COMMANDS:
        subparsers.add_parser(name, add_help=False)

    args, remainder = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    module_name, function_name = _COMMANDS[args.command]
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return int(function(remainder) or 0)


if __name__ == "__main__":
    sys.exit(main())
