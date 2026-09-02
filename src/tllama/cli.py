"""The `tllama` command.

Kept apart from main.py so that adding a subcommand does not mean editing
the module that also builds the FastAPI application. The entry point in
pyproject.toml has named tllama.main:main since before there was a main
to name; this is what it now reaches.
"""

from __future__ import annotations

import argparse
import logging
import sys

from typing import Any, Dict, List, Optional, Sequence

from tllama.config import ConfigError


def _rebuild_repo(args: argparse.Namespace) -> int:
    # Imported here rather than at module scope: `tllama --help` should
    # not have to load llama_cpp and construct a ModelManager first.
    from tllama.backend import model_manager

    model_manager.ensure_storage()

    results = model_manager.rebuild_repository(
        force=args.force,
        dry_run=args.dryrun,
    )

    return _report(results, models_dir=model_manager.models_dir, dry_run=args.dryrun)


def _report(results: List[Dict[str, Any]], *, models_dir, dry_run: bool) -> int:
    """Print what happened, grouped so the destructive part cannot hide.

    Overwrites are listed separately from creations on purpose. Forty new
    definitions and three replaced ones in one list means the three nobody
    can afford to miss are the three least likely to be read.
    """
    def shown(path) -> str:
        try:
            return str(path.relative_to(models_dir))
        except (ValueError, AttributeError):
            return str(path)

    created = [r for r in results if r["action"] == "create"]
    overwritten = [r for r in results if r["action"] == "overwrite"]
    skipped = [r for r in results if r["action"] == "skip"]

    verb = "would create" if dry_run else "created"
    for entry in created:
        print(f"{verb}: {shown(entry['toml'])}")

    if overwritten:
        print()
        verb = "WOULD OVERWRITE" if dry_run else "overwrote"
        for entry in overwritten:
            print(f"{verb}: {shown(entry['toml'])}")

    if skipped:
        print()
        for entry in skipped:
            print(f"skipped: {shown(entry['file'])} -- {entry['reason']}")

    print()
    print(
        f"{len(created)} to create, {len(overwritten)} to overwrite, "
        f"{len(skipped)} skipped"
        if dry_run
        else f"{len(created)} created, {len(overwritten)} overwritten, "
             f"{len(skipped)} skipped"
    )

    if dry_run:
        print("Nothing was written.")

    return 0


def _serve(args: argparse.Namespace) -> int:
    from tllama.main import start_server

    start_server()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tllama",
        description="TLlama server and model repository tools.",
    )

    subcommands = parser.add_subparsers(dest="command")

    serve = subcommands.add_parser("serve", help="Run the TLlama server.")
    serve.set_defaults(handler=_serve)

    rebuild = subcommands.add_parser(
        "rebuildrepo",
        help="Write the .toml definitions that make .gguf files into models.",
        description=(
            "A .gguf that no .toml points at is not a model: it is not "
            "listed and cannot be loaded. This writes the missing "
            "definitions, for a repository that predates them and equally "
            "for a file copied into Local/ by hand. Safe to run at any "
            "time: without --force it only adds files that were not there."
        ),
    )
    rebuild.add_argument(
        "--dryrun",
        action="store_true",
        help="Show what would happen and write nothing.",
    )
    rebuild.add_argument(
        "--force",
        action="store_true",
        help=(
            "Also replace definitions that already exist. Each replaced "
            "file is kept as a .toml.bak alongside it."
        ),
    )
    rebuild.set_defaults(handler=_rebuild_repo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    handler = getattr(args, "handler", None)
    if handler is None:
        # No subcommand runs the server, which is what this entry point
        # did before it had subcommands at all.
        handler = _serve

    # Every subcommand reads the environment, whether by starting the
    # server or by constructing the ModelManager, so this wraps all of
    # them rather than just serve. A bad variable is the operator's typo,
    # not a bug: a traceback would bury the one line that says which
    # variable and what it should look like.
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"tllama: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
