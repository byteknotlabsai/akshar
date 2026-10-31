"""Download a test set at its pinned revision and write it out as audio files plus manifest.jsonl.

Prepared layout, under $AKSHAR_DATA/eval/<set>/ (default ~/akshar-data):
    manifest.jsonl   one clip per line, in the source's order: id, audio (relative path), duration, text, ...
    audio/           each clip's audio bytes exactly as the source stores them (WAV or FLAC)
    source.json      repo, revision and SHA-256 of every downloaded file, clip count and hours
Clip order follows the source, so "the first N clips" means the same clips as in Udgam Labs' subsets.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from . import sets
from .audio import SAMPLE_RATE, sniff_extension
from .paths import EVAL_ROOT, RAW_ROOT, REPO_ROOT

SILENCE_SEED = 20261031  # fixed, so every run sees the same noise
SILENCE_SECONDS = 4.0
SILENCE_NOISE_STD = 0.003  # Udgam Labs: "very faint synthetic noise (Gaussian, standard deviation 0.003)"


def download(ts: sets.TestSet) -> list[Path]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    paths = []
    for name in ts.source.files:
        try:
            path = hf_hub_download(ts.source.repo, name, repo_type="dataset", revision=ts.source.revision,
                                   local_dir=RAW_ROOT / ts.name)
        except GatedRepoError:
            raise SystemExit(
                f"{ts.name}: {ts.source.repo} is gated. Accept its terms at "
                f"https://huggingface.co/datasets/{ts.source.repo} while logged in, then run `hf auth login` here "
                "and try again.") from None
        except HfHubHTTPError as error:
            raise SystemExit(f"{ts.name}: download of {name} failed: {error}") from None
        paths.append(Path(path))
    return paths


def _parquet_rows(paths, columns):
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        missing = [c for c in columns if c not in names]
        if missing:
            raise SystemExit(f"{path}: no column(s) {missing}; its columns are {names}")
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def _monsoon(paths):
    for row in _parquet_rows(paths, ["id", "audio", "lattice"]):
        lattice = row["lattice"]
        # `text` (the first written form of every span) is used only for CER and the blocklist; WER-style scoring of
        # Monsoon is OIWER against the whole lattice.
        yield {"id": row["id"], "blob": row["audio"]["bytes"],
               "text": " ".join(span[0] for span in lattice if span), "lattice": lattice}


def _fleurs(paths):
    tsv, archive = paths
    with open(tsv, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    with tarfile.open(archive) as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for sentence_id, file_name, raw, transcription, *_ in rows:
            # FLEURS has two reference columns: raw_transcription (as written) and transcription (FLEURS' own
            # normalisation, which keeps the danda). docs/eval.md records which one reproduces Shruti's numbers.
            yield {"id": Path(file_name).stem, "blob": tar.extractfile(members[file_name]).read(), "text": raw,
                   "fleurs_transcription": transcription, "fleurs_sentence_id": sentence_id}


def _librispeech(paths):
    for row in _parquet_rows(paths, ["id", "audio", "text"]):
        yield {"id": row["id"], "blob": row["audio"]["bytes"], "text": row["text"]}


def _mucs(paths):
    for row in _parquet_rows(paths, ["segment_id", "audio", "transcript"]):
        yield {"id": row["segment_id"], "blob": row["audio"]["bytes"], "text": row["transcript"]}


def _lahaja(paths):
    # Columns as the dataset card lists them at the pinned revision; the audio lives in `audio_filepath`.
    # `fname` names a whole recording, which is cut into many clips, so the clip id is the clip's own file name.
    # Lahaja has three transcript columns. All three are kept, and docs/eval.md records which one is the reference.
    for row in _parquet_rows(paths, ["audio_filepath", "fname", "text", "verbatim", "normalized"]):
        audio = row["audio_filepath"]
        yield {"id": Path(audio["path"]).stem, "blob": audio["bytes"], "text": row["text"],
               "lahaja_verbatim": row["verbatim"], "lahaja_normalized": row["normalized"],
               "lahaja_recording": row["fname"]}


def _svarah(paths):
    # Svarah has no id column; the clip's original file name, stored with its audio, serves as one.
    for i, row in enumerate(_parquet_rows(paths, ["audio_filepath", "text"])):
        audio = row["audio_filepath"]
        name = Path(audio["path"]).stem if audio.get("path") else f"svarah-{i:05d}"
        yield {"id": name, "blob": audio["bytes"], "text": row["text"]}


def _silence(paths):
    rng = np.random.default_rng(SILENCE_SEED)
    samples = int(SILENCE_SECONDS * SAMPLE_RATE)
    for i in range(100):
        if i < 50:
            clip, kind = np.zeros(samples, np.float32), "digital"
        else:
            clip, kind = (rng.standard_normal(samples) * SILENCE_NOISE_STD).astype(np.float32), "noise"
        buffer = io.BytesIO()
        sf.write(buffer, clip, SAMPLE_RATE, format="WAV", subtype="FLOAT")
        yield {"id": f"silence-{kind}-{i:03d}", "blob": buffer.getvalue(), "text": ""}


PREPARERS = {"monsoon": _monsoon, "fleurs_hi": _fleurs, "lahaja": _lahaja, "mucs": _mucs, "svarah": _svarah,
             "librispeech_clean": _librispeech, "silence": _silence}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare(name: str, force: bool = False) -> Path:
    ts = sets.get(name)
    out = EVAL_ROOT / name
    if (out / "manifest.jsonl").exists() and not force:
        print(f"{name}: already prepared in {out} (use --force to redo)")
        return out
    paths = download(ts) if ts.source else []
    partial = EVAL_ROOT / f".{name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    (partial / "audio").mkdir(parents=True)
    rows, seen = [], set()
    for item in PREPARERS[name](paths):
        blob = item.pop("blob")
        if item["id"] in seen:
            raise SystemExit(f"{name}: duplicate clip id {item['id']!r}")
        seen.add(item["id"])
        relative = "audio/" + re.sub(r"[^A-Za-z0-9._-]", "_", item["id"]) + sniff_extension(blob)
        (partial / relative).write_bytes(blob)
        info = sf.info(str(partial / relative))
        rows.append({"id": item.pop("id"), "audio": relative, "duration": round(info.duration, 4),
                     "sample_rate": info.samplerate, "channels": info.channels, **item})
    if len(rows) != ts.clips:
        print(f"WARNING {name}: {len(rows)} clips, but Udgam Labs report {ts.clips}")
    with open(partial / "manifest.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    source = {
        "set": name, "title": ts.title,
        "repo": ts.source.repo if ts.source else None, "revision": ts.source.revision if ts.source else None,
        "files": {str(p.relative_to(RAW_ROOT / name)): _sha256(p) for p in paths},
        "generated": None if ts.source else {"seed": SILENCE_SEED, "seconds": SILENCE_SECONDS,
                                             "noise_std": SILENCE_NOISE_STD},
        "clips": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 4),
        "prepared": datetime.now(timezone.utc).isoformat(timespec="seconds"), "akshar_commit": git_commit(),
    }
    (partial / "source.json").write_text(json.dumps(source, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    print(f"{name}: {len(rows)} clips, {source['hours']:.2f} h -> {out}")
    return out


def load_manifest(name: str) -> list[dict]:
    path = EVAL_ROOT / name / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"{name} is not prepared; run: python -m akshar.eval prepare {name}")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]
