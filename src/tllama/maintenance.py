"""Local repository maintenance for the daemon host."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List


def _rebuild_repo(args: argparse.Namespace) -> int:
    # Imported here rather than at module scope: `tllama --help` should
    # not have to construct the application and ModelManager first.
    from tllama.backend import model_manager

    # argparse cannot say "this one depends on that one", and letting it
    # pass silently would be the worse half of the same fault this whole
    # option exists to fix: a person who typed the destructive flag and
    # got no destruction, and no word about why.
    if args.overwrite_backups and not args.force:
        print(
            "tllamad: --overwrite-backups only means something with --force; "
            "on its own there is nothing to back up.",
            file=sys.stderr,
        )
        return 2

    model_manager.ensure_storage()

    results = model_manager.rebuild_repository(
        force=args.force,
        overwrite_backups=args.overwrite_backups,
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


def add_rebuild_parser(subcommands) -> None:
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
            "file is kept as a .toml.bak alongside it, and a file whose "
            ".toml.bak is already there is skipped rather than losing it."
        ),
    )
    rebuild.add_argument(
        "--overwrite-backups",
        action="store_true",
        help=(
            "Replace an existing .toml.bak instead of skipping the file. "
            "Says the previous backup is expendable; needs --force."
        ),
    )
    rebuild.set_defaults(handler=_rebuild_repo)
