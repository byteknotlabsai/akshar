"""The .cact models the server can run, and the one needle engine they share.

needle holds one model per process and is not thread-safe (needle/agent/whistle.py), so every call takes one lock.
Switching models reloads the weights, which takes about 0.1 s. That happens before the clock starts, so the speed
reported is decoding alone, timed as the harness times it (akshar.eval.run).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..eval.audio import SAMPLE_RATE
from ..eval.engine import sha256  # also sets the telemetry opt-out (D6) before needle is imported
from ..eval.paths import REPO_ROOT

MODELS_ROOT = REPO_ROOT / "models"
LANGUAGES = ("hi", "en", "auto")  # the protocol's two language flags, or detection by the engine
RECIPES = {"w4": "4-bit", "mixed2": "2-bit", "mixed2-enc4": "2-bit, encoder at 4 bits"}
# A train.json records only a model's last training stage, so models trained in several stages are named here.
ABOUT = {"shruti": "Udgam Labs' release, 4-bit",
         "akshar-v1": "AksharSTT v1 as released, 4-bit: v0, then 64,000 updates on 524 h and 2,400 on human transcripts",
         "akshar-v1-teacher": "akshar v1 before its last stage, 4-bit: v0, then 64,000 updates on 524 h"}


@dataclass
class Model:
    name: str  # the file's stem, e.g. akshar-v0
    path: Path
    file: str  # path shown to users, e.g. models/akshar-v0.cact
    bytes: int
    sha256: str
    about: str

    def info(self) -> dict:
        return {"name": self.name, "file": self.file, "mb": round(self.bytes / 1e6, 1), "about": self.about,
                "sha256": self.sha256}


def _about(path: Path) -> str:
    """What the model is, from the train.json saved beside it, if any."""
    if path.stem in ABOUT:
        return ABOUT[path.stem]
    try:
        run = json.loads(path.with_name(f"{path.stem}.train.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    recipe = RECIPES.get(run.get("recipe"), run.get("recipe") or "")
    steps = f"{run['steps']:,} updates" if run.get("steps") else ""
    return ", ".join(part for part in (recipe, steps) if part)


def find(root: Path = MODELS_ROOT) -> list[Model]:
    """Every .cact file under root, akshar's first and the largest first, skipping hidden folders (download caches)."""
    found = []
    for path in root.rglob("*.cact"):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        found.append(Model(path.stem, path, str(path.relative_to(root.parent)), path.stat().st_size, sha256(path),
                           _about(path)))
    found.sort(key=lambda m: ("akshar" not in m.name, -m.bytes, m.name))
    return found


_lock = threading.Lock()
_engines: dict[Path, object] = {}


def _whistle(model: Model):
    from needle import Whistle

    if model.path not in _engines:
        _engines[model.path] = Whistle(str(model.path))
    return _engines[model.path]


def load(model: Model) -> None:
    """Load a model now, so that a missing or incompatible engine shows when the server starts."""
    with _lock:
        _whistle(model)


def transcribe(model: Model, samples, language: str, keywords: list[str] | None = None, words: bool = True) -> dict:
    """Decode 16 kHz mono samples with the engine's defaults (5 beams).

    keywords are words or phrases for the engine to favour, such as names. Word timings and confidences come from the
    same decode and leave the text unchanged."""
    with _lock:
        whistle = _whistle(model)
        whistle._bind()  # load these weights now if another model ran last, so the clock times decoding alone
        tick = time.perf_counter()
        result = whistle.transcribe(samples, language=None if language == "auto" else language,
                                    keywords=keywords or None, word_timestamps=words)
        seconds = time.perf_counter() - tick
    audio_s = len(samples) / SAMPLE_RATE
    answer = {"text": result["text"], "model": model.name, "language": result.get("language"),
              "language_flag": language, "audio_s": round(audio_s, 2), "decode_s": round(seconds, 3),
              "x_real_time": round(audio_s / seconds, 1) if seconds > 0 else None}
    if words:
        answer["words"] = result.get("words") or []
    return answer
