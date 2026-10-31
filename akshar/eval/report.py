"""A Markdown table of every scored run, next to Udgam Labs' published numbers for Shruti."""
from __future__ import annotations

import json

from . import sets
from .paths import RUNS_ROOT
from .run import KNOWN_ENGINES


def _fmt(value, digits=2):
    return "–" if value is None else f"{value:.{digits}f}"


def _theirs(ts: sets.TestSet, build: str) -> float | None:
    """Udgam Labs' number for the same engine build: 3.3.2 is their build of 18 October, 3.3.3 their newest."""
    if build.startswith("3.3.3"):
        return ts.shruti_newest
    if build.startswith("3.3.2"):
        return ts.shruti_oct18
    return None


def report(include_partial: bool = False) -> str:
    lines = ["| Set | Model | Engine | Language | Clips | Ours | Udgam Labs, same build | Ours minus theirs | CER | "
             "Romanised WER | Speed (x real time) | Run |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(RUNS_ROOT.glob("*/*/scores.json")):
        scores = json.loads(path.read_text(encoding="utf-8"))
        meta = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        if not scores["complete"] and not include_partial:
            continue
        ts = sets.get(scores["set"])
        ours = scores["headline"]["value"]
        engine = meta["engine"]
        build = engine.get("build") or KNOWN_ENGINES.get(engine.get("library_sha256"), "unknown")
        # Udgam Labs' numbers are for the whole set with the protocol's language flag.
        comparable = not meta.get("limit") and scores["complete"] and scores["language"] == ts.language
        theirs = _theirs(ts, build) if comparable else None
        diff = _fmt(ours - theirs) if theirs is not None else "–"
        cer = _fmt(scores["cer"]["rate"]) if scores.get("cer") else "–"
        roman = _fmt(scores["romanised_wer"]["rate"]) if scores.get("romanised_wer") else "–"
        name = meta["model"].rsplit("/", 1)[-1]
        shown = ours if ts.scorer == "silence" else _fmt(ours)
        lines.append(f"| {ts.title} | {name} | {build.split(',')[0]} | {scores['language']} | {scores['clips']} | "
                     f"{shown} {scores['headline']['metric']} | {_fmt(theirs) if theirs is not None else '–'} | "
                     f"{diff} | {cer} | {roman} | {_fmt(scores['speed']['x_real_time'], 1)} | {path.parent.name} |")
    return "\n".join(lines)
