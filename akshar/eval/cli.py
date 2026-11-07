"""Command line for the evaluation harness: python -m akshar.eval <command> (or the akshar-eval script)."""
from __future__ import annotations

import argparse

from . import sets


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.eval", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("prepare", help="download test sets at their pinned revisions and write manifests")
    p.add_argument("sets", nargs="+", help=f"'all' or any of: {', '.join(sets.SETS)}")
    p.add_argument("--force", action="store_true", help="prepare again even if already prepared")

    p = commands.add_parser("run", help="decode a test set with a .cact model and score it")
    p.add_argument("--model", required=True, help="path to a .cact file")
    p.add_argument("--set", required=True, dest="set_name", choices=list(sets.SETS))
    p.add_argument("--language", choices=["hi", "en", "auto"], help="default: the protocol's flag for the set")
    p.add_argument("--limit", type=int, help="only the first N clips, in the source's order")
    p.add_argument("--out", help="run folder; an existing one is resumed")

    p = commands.add_parser("score", help="re-score run folders from their saved transcripts")
    p.add_argument("runs", nargs="+")
    p.add_argument("--first", type=int, help="score only the first N clips of the set; prints, writes nothing")

    p = commands.add_parser("report", help="table of all runs next to Udgam Labs' numbers")
    p.add_argument("--partial", action="store_true", help="include unfinished runs")

    p = commands.add_parser("blocklist", help="text and audio hashes of every prepared test clip")
    p.add_argument("sets", nargs="*", help="default: every prepared set except silence")

    p = commands.add_parser("dev", help="decode the dev subsets (first clips of every set) with a model")
    p.add_argument("model", help="path to a .cact file")
    p.add_argument("--name", required=True, help="label for this model in dev tables")

    p = commands.add_parser("dev-table", help="dev-subset scores of named models beside Shruti's")
    p.add_argument("names", nargs="*")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .prepare import prepare

        names = list(sets.SETS) if args.sets == ["all"] else args.sets
        for name in names:
            prepare(name, force=args.force)
    elif args.command == "run":
        from .run import run

        run(args.model, args.set_name, language=args.language, limit=args.limit, out=args.out)
    elif args.command == "score":
        import json

        from .run import score_run

        for run_dir in args.runs:
            print(run_dir, json.dumps(score_run(run_dir, first=args.first)["headline"], ensure_ascii=False))
    elif args.command == "report":
        from .report import report

        print(report(include_partial=args.partial))
    elif args.command == "blocklist":
        from .blocklist import build

        build(args.sets or None)
    elif args.command == "dev":
        from .dev import run_dev

        run_dev(args.model, args.name)
    elif args.command == "dev-table":
        from .dev import table

        print(table(args.names))
