"""Dev subsets: the first clips of every test set, for comparing training runs cheaply.

2-bit files decode slowly in the x86 engine (about 17 times real time), so a full seven-set evaluation of one model
takes about two hours. The dev subsets take about twelve minutes. They are the first N clips in source order, the
same clips Udgam Labs' subset scores use, and Shruti's scores on them come from our existing full runs by
re-scoring, with no new decoding. Final numbers always come from full sets.

    python -m akshar.eval dev runs/train/a2/model-mixed2.cact --name a2
    python -m akshar.eval dev-table a0 a1 a2
"""
from __future__ import annotations

import json
from pathlib import Path

from . import sets
from .paths import RUNS_ROOT

DEV_FIRST = {"monsoon": 300, "fleurs_hi": 200, "lahaja": 300, "mucs": 300, "svarah": 300, "librispeech_clean": 300,
             "silence": 100}


def _dir(name: str, set_name: str) -> Path:
    return RUNS_ROOT / "dev" / name / set_name


def run_dev(model: str, name: str) -> None:
    from .run import run

    for set_name, first in DEV_FIRST.items():
        out = _dir(name, set_name)
        if (out / "scores.json").exists() and json.loads((out / "scores.json").read_text())["complete"]:
            continue
        run(model, set_name, limit=first, out=str(out))


def shruti_reference() -> dict[str, float]:
    """Shruti 4-bit on the dev subsets, re-scored from its full protocol runs."""
    from .run import score_run

    out = {}
    for set_name, first in DEV_FIRST.items():
        runs = sorted(p.parent for p in (RUNS_ROOT / set_name).glob("shruti_*/scores.json")
                      if json.loads(p.read_text())["language"] == sets.get(set_name).language
                      and json.loads(p.read_text())["complete"]
                      and json.loads((p.parent / "run.json").read_text())["limit"] is None)
        if runs:
            out[set_name] = score_run(runs[0], first=min(first, sets.get(set_name).clips))["headline"]["value"]
    return out


def table(names: list[str]) -> str:
    reference = shruti_reference()
    header = "| Model | " + " | ".join(f"{s} (first {n})" for s, n in DEV_FIRST.items()) + " | Mean of WER sets |"
    lines = [header, "|" + "---|" * (len(DEV_FIRST) + 2)]

    def row(label, values):
        wer_sets = [v for s, v in values.items() if v is not None and sets.get(s).scorer != "silence"]
        mean = sum(wer_sets) / len(wer_sets) if len(wer_sets) == len(DEV_FIRST) - 1 else None
        cells = ["–" if values.get(s) is None else (str(int(values[s])) if sets.get(s).scorer == "silence"
                                                   else f"{values[s]:.2f}") for s in DEV_FIRST]
        return f"| {label} | " + " | ".join(cells) + f" | {'–' if mean is None else f'{mean:.2f}'} |"

    lines.append(row("Shruti 4-bit (released)", reference))
    for name in names:
        values = {}
        for set_name in DEV_FIRST:
            path = _dir(name, set_name) / "scores.json"
            if path.exists() and json.loads(path.read_text())["complete"]:
                values[set_name] = json.loads(path.read_text())["headline"]["value"]
        lines.append(row(name, values))
    return "\n".join(lines)
