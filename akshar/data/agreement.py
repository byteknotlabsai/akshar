"""Keep the labelled clips whose transcript agrees with the dataset's own text, for sources whose text is a script
rather than a verbatim transcript: Shrutilipi's clips are cut from news bulletins and matched to the bulletin's
script, which the reader did not always follow. The label audit of 2026-11-08 (D29) found akshar v2's transcript right
in 90% of the clips where it disagreed with the script on fewer than 5% of the words, in 42% between 5 and 15%, and
in 15% between 15 and 30%, so training on the agreeing clips keeps the good labels and drops most bad ones.

    python -m akshar.data.agreement shrutilipi_hi_v2 shrutilipi_hi_v2c --labels akshar-v2 --max-wer 5 \\
        --out shrutilipi_hi_agree [--lexicon lexicon.json] [--report-only]

Each folder under $AKSHAR_DATA/train gives its manifest.jsonl and labels-<labels>.jsonl. A clip's disagreement is the
word error rate of its label against its script, both normalised; with a lexicon, the script first goes through
akshar.data.convert.to_mixed so that English words written in Devanagari compare in Latin, as the label writes them.
The clips at or below --max-wer go to $AKSHAR_DATA/train/<out>/ as manifest.jsonl and labels-mix.jsonl (`by` names the
labeller), with the audio paths unchanged (the audio stays where the source folders' shards put it, or on Drive). The
first folder that lists an id wins. agreement.json there (or standard output with --report-only) gives the clips and
hours per disagreement band, so the threshold can be chosen with the audit's table in hand.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..eval.normalise import normalise
from ..eval.paths import DATA_ROOT
from ..eval.score import word_counts
from .convert import Lexicon, load_lexicon, to_mixed

BANDS = ((0.0, "0"), (5.0, "<5"), (15.0, "5-15"), (30.0, "15-30"), (60.0, "30-60"), (float("inf"), ">60"))


def band(wer: float) -> str:
    """The audit's disagreement band a per-clip word error rate (percent) falls in: 0, <5, 5-15, 15-30, 30-60, >60."""
    for upper, name in BANDS:
        if wer <= upper if upper == 0.0 else wer < upper:
            return name
    return ">60"


def disagreement(script: str, label: str, source: str, lexicon: Lexicon | None) -> float:
    """Word error rate (percent) of a label against its script, both as `mixed`-style text where a lexicon is given."""
    reference = to_mixed(script, source, lexicon) if lexicon else None
    return word_counts(reference if reference is not None else normalise(script), normalise(label)).rate


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_agreeing(names: list[str], labels_name: str, max_wer: float, out: str | None,
                    lexicon: dict[str, str] | None = None) -> dict:
    """Write the agreeing clips of these source folders to one folder (unless out is None) and return the report."""
    table = Lexicon(lexicon) if lexicon else None
    seen: set[str] = set()
    kept_rows, kept_labels = [], []
    clips, hours, counts = Counter(), Counter(), Counter()
    for name in names:
        folder = DATA_ROOT / "train" / name
        labels = {row["id"]: row for row in _read_jsonl(folder / f"labels-{labels_name}.jsonl")}
        for row in _read_jsonl(folder / "manifest.jsonl"):
            if row["id"] in seen:
                counts["duplicate"] += 1
                continue
            label = labels.get(row["id"])
            if label is None:
                counts["unlabelled"] += 1
                continue
            seen.add(row["id"])
            script = row.get("text") or ""
            if not script.strip() or not label["teacher"].strip():
                counts["empty"] += 1
                continue
            wer = disagreement(script, label["teacher"], row.get("source") or name, table)
            clips[band(wer)] += 1
            hours[band(wer)] += row["duration"] / 3600
            if wer <= max_wer:
                kept_rows.append(row)
                kept_labels.append({"id": row["id"], "teacher": label["teacher"], "language": label["language"],
                                    "by": labels_name})
    report = {"sources": names, "labels": f"labels-{labels_name}.jsonl", "max_wer": max_wer,
              "lexicon_words": len(lexicon) if lexicon else 0, "kept": len(kept_rows),
              "kept_hours": round(sum(r["duration"] for r in kept_rows) / 3600, 1),
              "bands": {name: {"clips": clips[name], "hours": round(hours[name], 1)} for _, name in BANDS},
              "skipped": dict(sorted(counts.items()))}
    if out is not None:
        folder = DATA_ROOT / "train" / out
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept_rows),
                                               encoding="utf-8")
        (folder / "labels-mix.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept_labels),
                                                 encoding="utf-8")
        (folder / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.agreement", description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="+", metavar="source", help="folders under $AKSHAR_DATA/train")
    parser.add_argument("--labels", required=True, metavar="NAME", help="read labels-NAME.jsonl (e.g. akshar-v2)")
    parser.add_argument("--max-wer", type=float, default=5.0, help="keep clips at or below this percent (default 5)")
    parser.add_argument("--out", metavar="FOLDER", help="write manifest.jsonl and labels-mix.jsonl under train/FOLDER")
    parser.add_argument("--lexicon", type=Path, help="convert the scripts' Devanagari English with this lexicon first")
    parser.add_argument("--report-only", action="store_true", help="print the bands and write nothing")
    args = parser.parse_args(argv)
    if not args.report_only and not args.out:
        parser.error("--out is needed unless --report-only")
    lexicon = load_lexicon(args.lexicon) if args.lexicon else None
    report = filter_agreeing(args.sources, args.labels, args.max_wer, None if args.report_only else args.out, lexicon)
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
