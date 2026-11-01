"""A training manifest from FLEURS Hindi's train split (licence ledger: admit, caveat).

Only clips whose normalised transcript is pure Devanagari are kept. Those are already in the `mixed` training
style (CLAUDE.md) without a converter: no digits, no Latin-script words, no punctuation. Every clip is checked
against the test-set blocklist by text and by audio.

    python -m akshar.data.fleurs --hours 3

Output: $AKSHAR_DATA/train/fleurs_hi_train/{manifest.jsonl, clips/, summary.json}. Manifest lines follow needle's
fine-tuning format: {"audio": "clips/<id>.wav", "text": ..., "language": "hi", "duration": ...}.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
from pathlib import Path

from ..eval import audio
from ..eval.blocklist import Blocklist
from ..eval.normalise import normalise
from ..eval.paths import DATA_ROOT

REPO, REVISION = "google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
DEVANAGARI_ONLY = re.compile(r"^[ऀ-ॿ ]+$")


def build(hours: float, out: Path) -> dict:
    from huggingface_hub import hf_hub_download

    raw = DATA_ROOT / "raw" / "fleurs_hi_train"
    tsv, archive = (Path(hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=raw))
                    for f in ("data/hi_in/train.tsv", "data/hi_in/audio/train.tar.gz"))
    with open(tsv, encoding="utf-8", newline="") as handle:
        rows = {r[1]: r for r in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)}
    blocklist = Blocklist()
    (out / "clips").mkdir(parents=True, exist_ok=True)
    counts = {"kept": 0, "not_devanagari": 0, "blocked_text": 0, "blocked_audio": 0}
    seconds = 0.0
    with tarfile.open(archive) as tar, open(out / "manifest.jsonl", "w", encoding="utf-8") as manifest:
        for member in tar:  # archive order, streamed: the 1.3 GB file is never read whole
            name = Path(member.name).name
            if not member.isfile() or name not in rows:
                continue
            text = normalise(rows[name][2])
            if not DEVANAGARI_ONLY.match(text):
                counts["not_devanagari"] += 1
                continue
            blob = tar.extractfile(member).read()
            path = out / "clips" / name
            path.write_bytes(blob)
            samples = audio.load(path)
            if len(samples) > 30 * audio.SAMPLE_RATE:  # longer than the model's window
                counts["over_30s"] = counts.get("over_30s", 0) + 1
                path.unlink()
                continue
            reason = blocklist.blocks(text, samples)
            if reason:
                counts[f"blocked_{reason}"] += 1
                path.unlink()
                continue
            duration = len(samples) / audio.SAMPLE_RATE
            manifest.write(json.dumps({"id": f"fleurs_hi_train/{Path(name).stem}", "audio": f"clips/{name}",
                                       "duration": round(duration, 3), "text": text, "language": "hi",
                                       "source": "fleurs_hi_train"}, ensure_ascii=False) + "\n")
            counts["kept"] += 1
            seconds += duration
            if seconds >= hours * 3600:
                break
    summary = {"source": f"{REPO}@{REVISION[:7]} hi_in train", **counts, "hours": round(seconds / 3600, 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.fleurs", description=__doc__.split("\n\n")[0])
    parser.add_argument("--hours", type=float, default=3.0, help="stop after this much kept audio")
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "train" / "fleurs_hi_train")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.hours, args.out)))


if __name__ == "__main__":
    main()
