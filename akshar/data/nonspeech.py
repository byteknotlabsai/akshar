"""Synthetic non-speech training clips with empty targets: silence and noise, so a model learns to write nothing.

    python -m akshar.data.nonspeech            # 600 clips into $AKSHAR_DATA/train/nonspeech

Shruti trained with silence and synthetic-noise clips whose transcripts are empty, and writes nothing on the harness's
silent clips. Speech datasets contain no such clips, so a model trained on them alone is never shown that silence
means "write nothing, then stop" (D22).

Four kinds, a quarter each: digital silence, white noise, pink noise (like room tone), and 50 Hz mains hum with its
harmonics over faint white noise. Noise is scaled to an RMS level drawn from -70 to -40 dB relative to full scale.
Lengths are drawn from 1 to 20 s and are never exactly 4 s, the length of every clip in the harness's silence set,
so no training clip can share a test clip's audio hash. A fixed seed makes every machine generate the same clips.
Half the clips of each kind carry the language tag hi, half en.

Output, under $AKSHAR_DATA/train/nonspeech/ (rewritten on every run):
    manifest.jsonl       id, audio, duration, text (empty), language, source, kind, level_db
    labels-shruti.jsonl  each clip's training target: empty by construction, marked "by": "construction"
    clips/               16 kHz mono 16-bit WAV
    summary.json         clips and hours per kind, and the settings
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ..eval.audio import SAMPLE_RATE
from ..eval.paths import DATA_ROOT

SEED = 20261104
CLIPS = 600
KINDS = ("silence", "white", "pink", "hum")
LANGUAGES = ("hi", "en")
LEVELS_DB = (-70.0, -40.0)
SECONDS = (1.0, 20.0)
EVAL_SECONDS = 4.0  # akshar.eval.prepare: every silence test clip is this long
MAINS_HZ = 50.0  # India's mains frequency


def _shape(rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
    if kind == "white":
        return rng.standard_normal(n)
    if kind == "pink":  # 1/f power: divide white noise's spectrum by the square root of frequency
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
        freqs[0] = freqs[1]  # no division by zero at DC
        pink = np.fft.irfft(spectrum / np.sqrt(freqs), n)
        return pink - pink.mean()
    if kind == "hum":  # the fundamental and three harmonics at falling amplitude, over faint white noise
        t = np.arange(n) / SAMPLE_RATE
        phases = rng.uniform(0, 2 * np.pi, 4)
        hum = sum(np.sin(2 * np.pi * MAINS_HZ * k * t + phases[k - 1]) / k for k in range(1, 5))
        return hum + 0.1 * rng.standard_normal(n)
    raise ValueError(kind)


def make_clip(rng: np.random.Generator, kind: str, n: int) -> tuple[np.ndarray, float | None]:
    """n samples of one kind of non-speech, and its RMS level in dB (None for digital silence)."""
    if kind == "silence":
        return np.zeros(n, np.float32), None
    level = round(float(rng.uniform(*LEVELS_DB)), 1)
    shaped = _shape(rng, kind, n)
    scaled = shaped / np.sqrt(np.mean(shaped ** 2)) * 10 ** (level / 20)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32), level


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def generate(out: Path, clips: int = CLIPS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows, labels = [], []
    for i in range(clips):
        kind, language = KINDS[i % len(KINDS)], LANGUAGES[(i // len(KINDS)) % len(LANGUAGES)]
        seconds = round(float(rng.uniform(*SECONDS)), 2)
        if seconds == EVAL_SECONDS:
            seconds += 0.01
        n = round(seconds * SAMPLE_RATE)
        samples, level = make_clip(rng, kind, n)
        name = f"{kind}-{i:04d}"
        sf.write(out / "clips" / f"{name}.wav", samples, SAMPLE_RATE, subtype="PCM_16")
        clip_id = f"nonspeech/{name}"
        rows.append({"id": clip_id, "audio": f"clips/{name}.wav", "duration": n / SAMPLE_RATE, "text": "",
                     "language": language, "source": "nonspeech", "kind": kind, "level_db": level})
        labels.append({"id": clip_id, "teacher": "", "language": language, "by": "construction"})
    _write_jsonl(out / "manifest.jsonl", rows)
    _write_jsonl(out / "labels-shruti.jsonl", labels)
    per_kind = {k: {"clips": sum(r["kind"] == k for r in rows),
                    "hours": round(sum(r["duration"] for r in rows if r["kind"] == k) / 3600, 3)} for k in KINDS}
    summary = {"clips": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 3), "kinds": per_kind,
               "seed": seed, "levels_db": LEVELS_DB, "seconds": SECONDS, "never_seconds": EVAL_SECONDS}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.nonspeech", description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", type=int, default=CLIPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "train" / "nonspeech")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.out, args.clips, args.seed)))


if __name__ == "__main__":
    main()
