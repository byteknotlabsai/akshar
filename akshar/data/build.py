"""Build a training set from one source: download a shard, keep its usable clips as 16 kHz FLAC, delete the shard.

    python -m akshar.data.build rural_women_hi --hours 40
    python -m akshar.data.build indicvoices_hi --hours 40 --fraction 0.5

Shards are visited in a spread-out order (sources.spread), so a partial build still covers the whole dataset.
`--fraction` keeps a fixed pseudo-random share of each shard's clips (by a hash of the clip's name), so the same
hours come from more shards and more speakers. A clip is kept if it is 0.5 to 30 s long and matches no test clip by text or audio (the blocklist). Each shard's
parquet file is deleted once processed: the sources stay on Hugging Face at pinned revisions, and only the compact
copy lives on disk. Interrupted builds resume from the last finished shard.

Output, under $AKSHAR_DATA/train/<source>/:
    manifest.jsonl   one clip per line: id, audio (relative path), duration, text (reference), language, source
    blocked.jsonl    every clip the blocklist removed: id, reason, normalised text
    clips/<shard>/   16 kHz mono FLAC
    progress.json    finished shards and running counts
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from ..eval.audio import SAMPLE_RATE, audio_sha256
from ..eval.blocklist import Blocklist
from ..eval.normalise import normalise
from ..eval.paths import DATA_ROOT
from ..files import repair_jsonl
from . import sources

MIN_SECONDS, MAX_SECONDS = 0.5, 30.0


def _decode(blob: bytes) -> np.ndarray:
    data, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    mono = data.mean(axis=1, dtype=np.float32)
    if rate != SAMPLE_RATE and len(mono):
        import soxr

        mono = soxr.resample(mono, rate, SAMPLE_RATE, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def _download(src: sources.Source, shard: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(src.repo, shard, repo_type="dataset", revision=src.revision,
                                local_dir=DATA_ROOT / "raw" / src.name))


def _repair_manifest(out: Path, done: list[str]) -> None:
    """Keep only clips from shards the progress file records as finished: a build killed between writing a
    shard's lines and recording the shard would otherwise list that shard twice after resuming."""
    manifest = out / "manifest.jsonl"
    repair_jsonl(manifest)
    if not manifest.exists():
        return
    finished = {Path(s).stem for s in done}
    lines = manifest.read_text(encoding="utf-8").splitlines()
    keep = [line for line in lines if json.loads(line)["audio"].split("/")[1] in finished]
    if len(keep) != len(lines):
        manifest.write_text("".join(line + "\n" for line in keep), encoding="utf-8")


def _sampled(key: str, fraction: float) -> bool:
    return fraction >= 1 or int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def build(name: str, hours: float, out: Path | None = None, fraction: float = 1.0) -> dict:
    import pyarrow.parquet as pq

    src = sources.get(name)
    out = out or DATA_ROOT / "train" / name
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "source": f"{src.repo}@{src.revision[:7]}", "ledger": src.ledger, "done": [], "kept": 0, "seconds": 0.0,
        "too_short": 0, "too_long": 0, "blocked_text": 0, "blocked_audio": 0, "unreadable": 0,
        "fraction": fraction}
    if progress.get("fraction", 1.0) != fraction:
        raise SystemExit(f"{out} was built with --fraction {progress.get('fraction', 1.0)}; use the same value")
    _repair_manifest(out, progress["done"])
    blocklist = Blocklist()
    order = [src.shards[i] for i in sources.spread(len(src.shards)) if src.shards[i] not in progress["done"]]
    columns = [c for c in (src.audio, src.text, src.id) if c]
    blocked_log = open(out / "blocked.jsonl", "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=1) as prefetch, open(out / "manifest.jsonl", "a", encoding="utf-8") as manifest:
        pending = prefetch.submit(_download, src, order[0]) if order and progress["seconds"] < hours * 3600 else None
        for k, shard in enumerate(order):
            if pending is None:
                break
            path = pending.result()
            more = k + 1 < len(order) and progress["seconds"] < hours * 3600
            pending = prefetch.submit(_download, src, order[k + 1]) if more else None
            shard_id = Path(shard).stem
            (out / "clips" / shard_id).mkdir(parents=True, exist_ok=True)
            lines = []
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    record = row[src.audio] or {}
                    key = str(row[src.id]) if src.id else str(record.get("path") or "")
                    if not _sampled(key, fraction):
                        continue
                    try:
                        samples = _decode(record["bytes"])
                    except Exception:  # a corrupt or empty clip is skipped and counted, never fatal
                        progress["unreadable"] += 1
                        continue
                    seconds = len(samples) / SAMPLE_RATE
                    if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
                        progress["too_short" if seconds < MIN_SECONDS else "too_long"] += 1
                        continue
                    text = row[src.text] or ""
                    reason = ("text" if blocklist.text_blocked(text) else
                              "audio" if audio_sha256(samples) in blocklist.audios else None)
                    if reason:
                        progress[f"blocked_{reason}"] += 1
                        blocked_log.write(json.dumps({"key": key, "reason": reason, "text": normalise(text)},
                                                     ensure_ascii=False) + "\n")
                        continue
                    clip_id = str(row[src.id]) if src.id else Path(record.get("path") or "").stem
                    if not clip_id:
                        clip_id = f"{shard_id}-{progress['kept']}"
                    relative = f"clips/{shard_id}/{clip_id}.flac"
                    sf.write(out / relative, samples, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                    lines.append(json.dumps({"id": f"{name}/{clip_id}", "audio": relative,
                                             "duration": round(seconds, 3), "text": text,
                                             "language": src.language, "source": name}, ensure_ascii=False))
                    progress["kept"] += 1
                    progress["seconds"] += seconds
            manifest.write("".join(line + "\n" for line in lines))
            manifest.flush()
            progress["done"].append(shard)
            progress_path.write_text(json.dumps(progress, indent=1) + "\n")
            os.remove(path)  # the source stays on Hugging Face at its pinned revision
            print(f"{name}: {shard} done, {progress['kept']} clips, {progress['seconds'] / 3600:.2f} h", flush=True)
            if progress["seconds"] >= hours * 3600:
                break
        if pending is not None:  # a prefetched shard we no longer need
            os.remove(pending.result())
    blocked_log.close()
    return {k: v for k, v in progress.items() if k != "done"} | {"shards": len(progress["done"])}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.build", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", choices=list(sources.SOURCES))
    parser.add_argument("--hours", type=float, required=True, help="stop after the shard that reaches this")
    parser.add_argument("--fraction", type=float, default=1.0, help="share of each shard's clips to keep")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source, args.hours, fraction=args.fraction)))


if __name__ == "__main__":
    main()
