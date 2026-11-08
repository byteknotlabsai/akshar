"""Build a training source from a Common Voice archive downloaded from Mozilla Data Collective (D26), and keep it on
Google Drive shard by shard, as akshar.data.stage does for Hugging Face sources (D14).

    python -m akshar.data.commonvoice cv-corpus-27.0-hi.tar.gz --name cv_hi --language hi \\
        --remote gdrive:akshar/p3 --rclone rclone --rclone-config ~/rclone.conf

Common Voice has no Hugging Face copy any more, so its archives (MP3 clips with TSV metadata) come from the owner's
Mozilla Data Collective account. The first pass over the archive reads the clip lists: by default train.tsv and
dev.tsv, so the test split stays out as a possible benchmark; an archive without them (a curated accent subset) uses
every TSV that lists `path` and `sentence`, except test, invalidated, other and reported. The second pass decodes each
listed clip to 16 kHz mono FLAC with akshar.data.build's checks: 0.5 to 30 s, and the blocklist of test clips by
normalised text and by audio hash (D17). Clips go into shards of --shard clips (clips/cv-00000/...), and each full
shard is packed and uploaded by akshar.data.stage's Stager, then deleted, so the disk holds about one shard at a time.

Writes, under $AKSHAR_DATA/train/<name>/: manifest.jsonl (the `sentence` is the text), progress.json, blocked.jsonl
and summary.json, all uploaded beside the shards. Not resumable: a failed build starts again in an empty folder.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import threading
import time
from pathlib import Path

import soundfile as sf

from ..eval.audio import SAMPLE_RATE, audio_sha256
from ..eval.blocklist import Blocklist
from ..eval.normalise import normalise
from ..eval.paths import DATA_ROOT
from .build import MAX_SECONDS, MIN_SECONDS, _decode
from .stage import Stager

SKIP_LISTS = ("test", "invalidated", "other", "reported", "clip_durations", "unvalidated", "validated_sentences")


def read_lists(archive: Path, wanted: tuple[str, ...]) -> dict[str, str]:
    """Clip file name -> sentence, from the archive's TSVs: those named in `wanted`, else every clip list."""
    tables: dict[str, list[dict]] = {}
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".tsv")):
                continue
            stem = Path(member.name).stem
            text = tar.extractfile(member).read().decode("utf-8")
            rows = list(csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))
            if rows and "path" in rows[0] and "sentence" in rows[0]:
                tables[stem] = rows
    chosen = [s for s in wanted if s in tables] or [s for s in tables if not any(k in s for k in SKIP_LISTS)]
    if not chosen:
        raise SystemExit(f"{archive}: no clip list among {sorted(tables)}")
    print(f"clip lists {chosen} of {sorted(tables)}", flush=True)
    return {Path(row["path"]).name: row["sentence"] for s in chosen for row in tables[s]}


def build(archive: Path, name: str, language: str, out: Path, shard_size: int, wanted: tuple[str, ...]) -> dict:
    sentences = read_lists(archive, wanted)
    blocklist = Blocklist()
    progress = {"source": f"Common Voice, {archive.name}", "ledger": "admit (D26)", "done": [], "kept": 0,
                "seconds": 0.0, "listed": len(sentences), "too_short": 0, "too_long": 0, "blocked_text": 0,
                "blocked_audio": 0, "unreadable": 0}
    progress_path = out / "progress.json"
    blocked_log = open(out / "blocked.jsonl", "a", encoding="utf-8")
    manifest = open(out / "manifest.jsonl", "a", encoding="utf-8")
    shard, lines, in_shard = 0, [], 0

    def close_shard():
        nonlocal shard, lines, in_shard
        manifest.write("".join(line + "\n" for line in lines))
        manifest.flush()
        progress["done"].append(f"cv-{shard:05d}")
        progress_path.write_text(json.dumps(progress, indent=1) + "\n")
        print(f"{name}: cv-{shard:05d} done, {progress['kept']} clips, {progress['seconds'] / 3600:.2f} h", flush=True)
        shard, lines, in_shard = shard + 1, [], 0

    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            sentence = sentences.get(Path(member.name).name) if member.isfile() else None
            if sentence is None:
                continue
            try:
                samples = _decode(tar.extractfile(member).read())
            except Exception:  # a corrupt clip is skipped and counted, never fatal
                progress["unreadable"] += 1
                continue
            seconds = len(samples) / SAMPLE_RATE
            if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
                progress["too_short" if seconds < MIN_SECONDS else "too_long"] += 1
                continue
            reason = ("text" if blocklist.text_blocked(sentence) else
                      "audio" if audio_sha256(samples) in blocklist.audios else None)
            if reason:
                progress[f"blocked_{reason}"] += 1
                blocked_log.write(json.dumps({"key": member.name, "reason": reason, "text": normalise(sentence)},
                                             ensure_ascii=False) + "\n")
                continue
            clip_id = Path(member.name).stem
            relative = f"clips/cv-{shard:05d}/{clip_id}.flac"
            (out / relative).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out / relative, samples, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            lines.append(json.dumps({"id": f"{name}/{clip_id}", "audio": relative, "duration": round(seconds, 3),
                                     "text": sentence, "language": language, "source": name}, ensure_ascii=False))
            progress["kept"] += 1
            progress["seconds"] += seconds
            in_shard += 1
            if in_shard >= shard_size:
                close_shard()
    if lines:
        close_shard()
    blocked_log.close()
    manifest.close()
    return {k: v for k, v in progress.items() if k != "done"} | {"shards": len(progress["done"])}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.commonvoice", description=__doc__.split("\n\n")[0])
    parser.add_argument("archive", type=Path)
    parser.add_argument("--name", required=True, help="the source's name, e.g. cv_hi or cv_en_sa")
    parser.add_argument("--language", required=True, choices=["hi", "en"])
    parser.add_argument("--lists", nargs="+", default=["train", "dev"], help="TSV clip lists to take, by name")
    parser.add_argument("--shard", type=int, default=4000, help="clips per shard")
    parser.add_argument("--remote", help="rclone path that holds one folder per source; without it, nothing uploads")
    parser.add_argument("--rclone", default="rclone")
    parser.add_argument("--rclone-config")
    parser.add_argument("--bwlimit", default="15M")
    args = parser.parse_args(argv)
    out = DATA_ROOT / "train" / args.name
    if (out / "progress.json").exists():
        raise SystemExit(f"{out} already holds a build; remove it to build again")
    out.mkdir(parents=True, exist_ok=True)
    stager = worker = None
    if args.remote:
        rclone = [args.rclone, "--bwlimit", args.bwlimit, "--drive-chunk-size", "128M"]
        if args.rclone_config:
            rclone += ["--config", str(Path(args.rclone_config).expanduser())]
        stager = Stager(args.name, out, args.remote, rclone)
        worker = threading.Thread(target=stager.loop, daemon=True)
        worker.start()
    started = time.time()
    summary = build(args.archive, args.name, args.language, out, args.shard, tuple(args.lists))
    if stager is not None:
        stager.stop.set()
        worker.join()
        stager.stage_ready()
        stager.upload_lists()
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if stager is not None:
        stager._run("copyto", str(out / "summary.json"), f"{stager.remote}/summary.json")
    print(json.dumps({**summary, "staged_shards": len(stager.staged) if stager else 0,
                      "minutes": round((time.time() - started) / 60, 1)}))


if __name__ == "__main__":
    main()
