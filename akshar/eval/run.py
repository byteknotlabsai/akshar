"""Decode a prepared test set with a .cact model in the needle engine, then score it.

A run folder (runs/<set>/<model>_<language>[_first<N>]_<time>/) holds:
    hyps.jsonl   one line per clip: id, hyp (raw engine text), language, duration, decode_s
    run.json     what ran: model and engine hashes, settings, machine, memory
    scores.json  the scores, recomputable at any time with `python -m akshar.eval score <run folder>`
A run that stops part-way can be resumed with --out <run folder>; finished clips are skipped.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

from . import audio, sets
from .normalise import VERSION as NORMALISER_VERSION
from .paths import EVAL_ROOT, RUNS_ROOT
from .prepare import git_commit, load_manifest
from .score import score_clips

# D6: the cactus-needle package reports usage unless this is set before it is imported.
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# Engine libraries we have run, by SHA-256 (Linux x86_64). Udgam Labs' main table used the build of 18 October 2026,
# and their "newest build" numbers came from the build of 19 October (technical report, section 7.5).
KNOWN_ENGINES = {
    "87476ff70e80334d7c766dbf236b72f42876072df22014c6ec8a9b087b82b260": "3.3.3, needle3@d6c7d6a (2026-10-19)",
    "0c8f3970fab6ca96f59ba989377c7e61def64ee0e3b3970d7a06cdf93271d454": "3.3.2, needle3@3391a10 (2026-10-18)",
}


def engine_info() -> dict:
    """The engine library actually loaded. NEEDLE3_LIB_PATH overrides the build the package pins."""
    import needle
    from needle import _library_path
    from needle.agent import fetch

    library = _library_path(3)
    digest = _sha256(library)
    return {"package": needle.__version__, "pinned_engine": fetch.engine_version(3),
            "override": os.environ.get(fetch.lib_path_env(3)), "library_sha256": digest,
            "build": KNOWN_ENGINES.get(digest, "unknown")}


def run(model: str, set_name: str, language: str | None = None, limit: int | None = None, out: str | None = None,
        every: int = 100) -> Path:
    import psutil

    ts = sets.get(set_name)
    clips = load_manifest(set_name)
    if limit:
        clips = clips[:limit]
    # The protocol sets the language per set; "auto" leaves detection to the engine.
    language = language or ts.language
    flag = None if language == "auto" else language
    model_path = Path(model).resolve()
    if out:
        run_dir = Path(out)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        subset = f"_first{limit}" if limit else ""
        run_dir = RUNS_ROOT / set_name / f"{model_path.stem}_{language}{subset}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    hyps_path = run_dir / "hyps.jsonl"
    done = set()
    if hyps_path.exists():
        with open(hyps_path, encoding="utf-8") as handle:
            done = {json.loads(line)["id"] for line in handle}

    meta = {
        "set": set_name, "model": str(model_path), "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size, "language": language, "limit": limit, "clips_planned": len(clips),
        "decoding": "needle engine defaults (5 beams)", "normaliser": NORMALISER_VERSION,
        "engine": engine_info(), "akshar_commit": git_commit(),
        "machine": {"cpu": _cpu_name(), "logical_cpus": os.cpu_count(), "system": platform.platform()},
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "resumed_from": len(done),
    }
    previous = run_dir / "run.json"
    if done and previous.exists():
        # Resuming must not mix transcripts from a different model, set, language, subset or engine build.
        old = json.loads(previous.read_text(encoding="utf-8"))
        same = (old["set"], old["model_sha256"], old["language"], old.get("limit"),
                old["engine"]["library_sha256"]) == (set_name, meta["model_sha256"], language, limit,
                                                     meta["engine"]["library_sha256"])
        if not same:
            raise SystemExit(f"{run_dir} holds a run with different settings; use a new --out folder")
    previous.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    from needle import Whistle

    process = psutil.Process()
    rss_before = process.memory_info().rss
    whistle = Whistle(str(model_path))
    rss_loaded = process.memory_info().rss

    todo = [c for c in clips if c["id"] not in done]
    print(f"{set_name}: {len(todo)} clips to decode ({len(done)} already done) with {model_path.name}, "
          f"language {language} -> {run_dir}", flush=True)
    started = time.perf_counter()
    audio_s = decode_s = 0.0
    with open(hyps_path, "a", encoding="utf-8") as handle:
        for i, clip in enumerate(todo, 1):
            samples = audio.load(EVAL_ROOT / set_name / clip["audio"])
            tick = time.perf_counter()
            result = whistle.transcribe(samples, language=flag)
            seconds = time.perf_counter() - tick
            audio_s, decode_s = audio_s + clip["duration"], decode_s + seconds
            handle.write(json.dumps({"id": clip["id"], "hyp": result["text"], "language": result["language"],
                                     "duration": clip["duration"], "decode_s": round(seconds, 4)},
                                    ensure_ascii=False) + "\n")
            handle.flush()
            if i % every == 0 or i == len(todo):
                elapsed = time.perf_counter() - started
                print(f"  {i}/{len(todo)}  {elapsed:6.0f} s  {audio_s / max(decode_s, 1e-9):5.1f}x real time",
                      flush=True)

    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["memory_mb"] = {
        # Whole Python process, so these overstate the engine alone; Phase 4 measures devices separately.
        "process_peak": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "model_load_increase": round((rss_loaded - rss_before) / 2**20, 1),
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    scores = score_run(run_dir)
    print(json.dumps(scores["headline"], ensure_ascii=False))
    return run_dir


def score_run(run_dir: str | Path, first: int | None = None) -> dict:
    """Score a run folder and write scores.json. With `first`, score only the first N clips of the set (in the
    source's order, as Udgam Labs' subsets are) and write nothing."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    ts = sets.get(meta["set"])
    ordered = load_manifest(meta["set"])
    manifest = {c["id"]: c for c in ordered}
    with open(run_dir / "hyps.jsonl", encoding="utf-8") as handle:
        hyps = [json.loads(line) for line in handle]
    if first:
        wanted = {c["id"] for c in ordered[:first]}
        hyps = [h for h in hyps if h["id"] in wanted]
        if len(hyps) != first:
            raise SystemExit(f"{run_dir}: has {len(hyps)} of the first {first} clips")
    clips = [{**manifest[h["id"]], "hyp": h["hyp"]} for h in hyps]
    scores = score_clips(clips, ts.scorer, romanised=ts.romanised)
    audio_s = sum(h["duration"] for h in hyps)
    decode_s = sum(h["decode_s"] for h in hyps)
    if ts.scorer == "silence":
        headline = {"metric": "words written on silence", "value": scores.silence_words}
    elif ts.scorer == "oiwer":
        headline = {"metric": "OIWER %", "value": round(scores.oiwer.rate, 2)}
    else:
        headline = {"metric": "WER %", "value": round(scores.wer.rate, 2)}
    result = {
        "set": meta["set"], "clips": len(hyps), "complete": len(hyps) == meta["clips_planned"],
        "language": meta["language"], "normaliser": NORMALISER_VERSION, "headline": headline,
        # Plain WER only where the reference is one string; Monsoon's lattice is scored by OIWER alone.
        "wer": scores.wer.as_dict() if ts.scorer == "wer" else None,
        "cer": scores.cer.as_dict() if ts.cer else None,
        "oiwer": scores.oiwer.as_dict() if ts.scorer == "oiwer" else None,
        "romanised_wer": scores.roman.as_dict() if ts.romanised else None,
        "silence": {"words": scores.silence_words, "clips_with_words": scores.silence_clips_with_words}
        if ts.scorer == "silence" else None,
        "speed": {"audio_s": round(audio_s, 1), "decode_s": round(decode_s, 1),
                  "rtf": round(decode_s / audio_s, 4) if audio_s else None,
                  "x_real_time": round(audio_s / decode_s, 1) if decode_s else None},
        "languages_detected": dict(collections.Counter(h["language"] or "none" for h in hyps)),
    }
    if first:
        result["first"] = first
        result["complete"] = False
    else:
        (run_dir / "scores.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    return result
