"""Stored test results for the page, read from the harness's run folders (akshar.eval.run).

A results root holds <set>/<run>/{run.json, hyps.jsonl, scores.json}: runs/ for this machine's runs and
runs/server-full/ for full runs copied from the GPU server. A table never mixes roots, so the models in it were scored
on the same machine. Only complete protocol runs count: every clip of the set, decoded with the set's language flag.

One row per model file and engine build (by SHA-256), so a model scored under two folder names is one row. Per-clip
errors come from the harness's own scorers, so a run's errors summed over its clips are the errors in its scores.json.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..eval import sets
from ..eval.engine import KNOWN_ENGINES
from ..eval.normalise import normalise
from ..eval.paths import EVAL_ROOT, RUNS_ROOT
from ..eval.prepare import load_manifest
from ..eval.score import oiwer_counts, word_counts
from .align import text_alignment

SHORT = {"monsoon": "Monsoon", "fleurs_hi": "FLEURS Hindi", "lahaja": "Lahaja", "mucs": "MUCS", "svarah": "Svarah",
         "librispeech_clean": "LibriSpeech clean", "silence": "Silence"}
METRIC = {"wer": "WER", "oiwer": "OIWER", "silence": "words written"}
STAMP = re.compile(r"(_(hi|en|auto))?(_first\d+)?_\d{8}-\d{6}$")  # the suffix `python -m akshar.eval run` adds


@dataclass
class Run:
    key: str  # model and engine hashes: one table row per model file and engine build
    label: str
    folder: Path
    meta: dict  # run.json
    scores: dict  # scores.json


def complete_runs(root: Path, set_name: str) -> list[Run]:
    """The complete protocol runs of one set under a results root. If a model and engine build were scored more than
    once, the last folder by name counts (for timestamped folders, the latest)."""
    folder = root / set_name
    if not folder.is_dir():
        return []
    language = sets.SETS[set_name].language
    found: dict[str, Run] = {}
    for run_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        try:
            meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not scores.get("complete") or meta.get("limit") or meta.get("language") != language:
            continue
        key = f"{meta['model_sha256'][:12]}-{meta['engine']['library_sha256'][:12]}"
        found[key] = Run(key, STAMP.sub("", run_dir.name), run_dir, meta, scores)
    return list(found.values())


def engine_build(meta: dict) -> str:
    """The engine build a run used. Early run.json files lack the "build" field, so the library's hash decides."""
    engine = meta["engine"]
    return KNOWN_ENGINES.get(engine["library_sha256"]) or engine.get("build") or engine.get("engine") or "unknown"


def roots() -> dict[str, dict]:
    """Results roots holding at least one complete run: {name: {"path", "label"}}, the server's first."""
    found = {}
    for name, path, label in (("server", RUNS_ROOT / "server-full", "Server"), ("local", RUNS_ROOT, "This machine")):
        runs = [run for set_name in sets.SETS for run in complete_runs(path, set_name)]
        if runs:
            cpus = sorted({run.meta["machine"]["cpu"] for run in runs})
            found[name] = {"path": path, "label": f"{label}: {', '.join(cpus)}"}
    return found


def summary(root: Path) -> dict:
    """The full-test table of a root: one row per model and engine build, one cell per set, and the mean of the six
    speech sets where a row has all six. Also what the numbers depend on, to print beside them (CLAUDE.md)."""
    rows: dict[str, dict] = {}
    machines, engines, decodings, normalisers = set(), set(), set(), set()
    for set_name in sets.SETS:
        for run in complete_runs(root, set_name):
            meta = run.meta
            row = rows.setdefault(run.key, {
                "key": run.key, "label": run.label, "model": Path(meta["model"]).name,
                "model_sha256": meta["model_sha256"], "mb": round(meta["model_bytes"] / 1e6, 1),
                "engine": engine_build(meta), "cells": {}})
            row["cells"][set_name] = run.scores["headline"]["value"]
            machines.add(meta["machine"]["cpu"])
            engines.add(engine_build(meta))
            decodings.add(meta["decoding"])
            normalisers.add(run.scores["normaliser"])
    speech = [name for name, ts in sets.SETS.items() if ts.scorer != "silence"]
    for row in rows.values():
        values = [row["cells"][name] for name in speech if name in row["cells"]]
        row["mean"] = round(sum(values) / len(values), 2) if len(values) == len(speech) else None
    labels = [row["label"] for row in rows.values()]
    for row in rows.values():
        if labels.count(row["label"]) > 1:  # the same name scored with two engine builds
            row["label"] = f"{row['label']} ({row['engine'].split(',')[0]})"
    return {"sets": [{"name": name, "short": SHORT.get(name, ts.title), "title": ts.title, "metric": METRIC[ts.scorer],
                      "language": ts.language} for name, ts in sets.SETS.items()],
            "rows": list(rows.values()), "machines": sorted(machines), "engines": sorted(engines),
            "decoding": sorted(decodings), "normalisers": sorted(normalisers)}


_manifests: dict[str, dict[str, dict]] = {}
_hyps: dict[Path, tuple[int, dict[str, str]]] = {}
_tables: dict[tuple, dict] = {}


def manifest(set_name: str) -> dict[str, dict]:
    """A prepared test set's clips by id."""
    if set_name not in _manifests:
        _manifests[set_name] = {clip["id"]: clip for clip in load_manifest(set_name)}
    return _manifests[set_name]


def hyps(run: Run) -> dict[str, str]:
    """A run's raw transcripts by clip id, re-read whenever its hyps.jsonl changes."""
    path = run.folder / "hyps.jsonl"
    stamp = path.stat().st_mtime_ns
    if path not in _hyps or _hyps[path][0] != stamp:
        with open(path, encoding="utf-8") as handle:
            _hyps[path] = (stamp, {row["id"]: row["hyp"] for row in (json.loads(line) for line in handle if line.strip())})
    return _hyps[path][1]


def clip_counts(set_name: str, clip: dict, hyp: str) -> tuple[int, int]:
    """One clip's (errors, reference units) under its set's scorer: what akshar.eval.score sums over a set.
    On the silence set, the words written and 0."""
    scorer = sets.SETS[set_name].scorer
    if scorer == "silence":
        return len(normalise(hyp).split()), 0
    if scorer == "oiwer":
        counts = oiwer_counts(hyp, clip["lattice"])
    else:
        counts = word_counts(normalise(clip["text"]), normalise(hyp))
    return counts.errors, counts.units


def alignment(set_name: str, clip: dict, hyp: str) -> dict:
    """text_alignment for one test clip under its set's scorer. On Monsoon the alignment is the lattice scorer's own,
    against the reference spellings it matched; on the silence set every word written is an error."""
    scorer = sets.SETS[set_name].scorer
    if scorer == "silence":
        words = normalise(hyp).split()
        return {"metric": METRIC[scorer], "errors": len(words), "units": 0, "ops": [["i", "", w] for w in words]}
    if scorer == "oiwer":
        from voi_oiwer import oiwer

        result = oiwer(hyp, clip["lattice"], "hindi")  # called as akshar.eval.score.oiwer_counts calls it
        ops = [[op, "" if op == "i" else r, "" if op == "d" else h] for h, r, op in zip(*result[1:4])]
        return {"metric": METRIC[scorer], "errors": sum(result[4]), "units": result[5], "ops": ops}
    return text_alignment(clip["text"], hyp)


def audio_path(set_name: str, clip_id: str) -> Path | None:
    """A test clip's audio file, or None when the set has no such clip or its file is missing."""
    clip = manifest(set_name).get(clip_id)
    if clip is None:
        return None
    folder = (EVAL_ROOT / set_name).resolve()
    path = (folder / clip["audio"]).resolve()
    return path if folder in path.parents and path.is_file() else None


def clip_table(root: Path, set_name: str) -> dict:
    """Every clip of a set with each run's errors and reference units, for sorting and filtering on the page."""
    runs = complete_runs(root, set_name)
    labels = {row["key"]: row["label"] for row in summary(root)["rows"]}
    signature = (str(root), set_name, tuple((str(r.folder), (r.folder / "hyps.jsonl").stat().st_mtime_ns) for r in runs))
    if signature not in _tables:
        texts = [hyps(run) for run in runs]
        clips = []
        for clip in manifest(set_name).values():
            counts = [clip_counts(set_name, clip, t[clip["id"]]) if clip["id"] in t else (None, None) for t in texts]
            clips.append({"id": clip["id"], "duration": clip["duration"], "ref": clip.get("text", ""),
                          "e": [c[0] for c in counts], "u": [c[1] for c in counts]})
        ts = sets.SETS[set_name]
        _tables[signature] = {"set": set_name, "title": ts.title, "short": SHORT.get(set_name, ts.title),
                              "metric": METRIC[ts.scorer], "language": ts.language,
                              "runs": [{"key": run.key, "label": labels[run.key], "model_sha256": run.meta["model_sha256"]}
                                       for run in runs], "clips": clips}
    return _tables[signature]


def clip_detail(root: Path, set_name: str, clip_id: str) -> dict | None:
    """One clip: its reference and each run's transcript with its errors and alignment. None for an unknown id."""
    clip = manifest(set_name).get(clip_id)
    if clip is None:
        return None
    labels = {row["key"]: row["label"] for row in summary(root)["rows"]}
    runs = []
    for run in complete_runs(root, set_name):
        hyp = hyps(run).get(clip_id)
        if hyp is not None:
            runs.append({"key": run.key, "label": labels[run.key], "model_sha256": run.meta["model_sha256"],
                         "hyp": hyp, **alignment(set_name, clip, hyp)})
    ts = sets.SETS[set_name]
    return {"set": set_name, "title": ts.title, "id": clip_id, "duration": clip["duration"],
            "ref": clip.get("text", ""), "lattice": clip.get("lattice"), "metric": METRIC[ts.scorer],
            "language": ts.language, "runs": runs}
