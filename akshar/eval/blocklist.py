"""Hashes of every test clip, so that no test clip can enter training (CLAUDE.md, "Contamination").

For each clip: the SHA-256 of its normalised reference text and of its audio as 16 kHz mono 16-bit PCM. Training data
preparation (Phase 3) must drop any clip whose text hash or audio hash is in this list, including data added later.
The synthetic silence set is left out: it is not a recording, and training deliberately uses silence of its own.

Output: $AKSHAR_DATA/eval/blocklist.jsonl (one line per test clip) and blocklist.json (counts and settings).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import audio, sets
from .normalise import VERSION as NORMALISER_VERSION
from .normalise import normalise
from .paths import EVAL_ROOT
from .prepare import load_manifest

EXTRA_TEXT_FIELDS = ("fleurs_transcription", "lahaja_verbatim", "lahaja_normalized")  # other reference columns; hashed too


def text_sha256(text: str) -> str | None:
    """Hash of the normalised text, or None if nothing is left after normalising."""
    text = normalise(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


def build(names: list[str] | None = None) -> Path:
    names = names or [n for n, ts in sets.SETS.items()
                      if ts.scorer != "silence" and (EVAL_ROOT / n / "manifest.jsonl").exists()]
    out = EVAL_ROOT / "blocklist.jsonl"
    texts, audios, per_set = set(), set(), {}
    with open(out, "w", encoding="utf-8") as handle:
        for name in names:
            clips = load_manifest(name)
            for clip in clips:
                hashes = {text_sha256(clip.get(f) or "") for f in ("text",) + EXTRA_TEXT_FIELDS if f in clip}
                hashes.discard(None)
                audio_hash = audio.audio_sha256(audio.load(EVAL_ROOT / name / clip["audio"]))
                texts.update(hashes)
                audios.add(audio_hash)
                handle.write(json.dumps({"set": name, "id": clip["id"], "text_sha256": sorted(hashes),
                                         "audio_sha256": audio_hash}) + "\n")
            per_set[name] = len(clips)
            print(f"  {name}: {len(clips)} clips", flush=True)
    summary = {"sets": per_set, "unique_text_hashes": len(texts), "unique_audio_hashes": len(audios),
               "normaliser": NORMALISER_VERSION, "audio": "sha256 of 16 kHz mono int16 little-endian PCM",
               "built": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (EVAL_ROOT / "blocklist.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"blocklist: {sum(per_set.values())} clips, {len(texts)} text hashes, {len(audios)} audio hashes -> {out}")
    return out


class Blocklist:
    """Membership test against the built blocklist: a clip is blocked if its text or its audio matches a test clip."""

    def __init__(self, path: Path | None = None):
        path = path or EVAL_ROOT / "blocklist.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} does not exist; run: python -m akshar.eval blocklist")
        self.texts, self.audios = set(), set()
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                self.texts.update(row["text_sha256"])
                self.audios.add(row["audio_sha256"])

    def blocks(self, text: str, samples) -> str | None:
        """Why the clip is blocked ("text" or "audio"), or None. `samples` is 16 kHz mono float audio."""
        if text_sha256(text) in self.texts:
            return "text"
        if audio.audio_sha256(samples) in self.audios:
            return "audio"
        return None
