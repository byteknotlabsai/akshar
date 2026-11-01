"""Label a training set with a teacher model's own transcripts, decoded by the engine exactly as the eval harness does.

    python -m akshar.data.label librispeech_100 --model models/shruti/shruti.cact

Track A (Phase 2) trains a 2-bit copy of Shruti, so its targets are what Shruti writes, not each dataset's own
transcripts. Those follow other conventions (English words in Devanagari, digits, noise tags) that only Phase 3's
converters will turn into our `mixed` style. Shruti's output already is that style.

Writes, next to the source's manifest: labels-<model>.jsonl (id, teacher, language, decode_s) and labels-<model>.json
(model and engine hashes). Resumable: clips already labelled are skipped. Runs on the CPU, beside GPU training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..eval import audio
from ..files import repair_jsonl
from ..eval.paths import DATA_ROOT
from ..eval.run import _sha256, engine_info


def label(name: str, model: Path, every: int = 1000) -> dict:
    from needle import Whistle

    folder = DATA_ROOT / "train" / name
    with open(folder / "manifest.jsonl", encoding="utf-8") as handle:
        clips = [json.loads(line) for line in handle]
    out = folder / f"labels-{model.stem}.jsonl"
    done = set()
    repair_jsonl(out)  # a killed run can leave half a line
    if out.exists():
        with open(out, encoding="utf-8") as handle:
            done = {json.loads(line)["id"] for line in handle}
    meta = {"model": str(model), "model_sha256": _sha256(model), "engine": engine_info(),
            "decoding": "needle engine defaults (5 beams), language flag from the manifest"}
    (folder / f"labels-{model.stem}.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    whistle = Whistle(str(model))
    todo = [c for c in clips if c["id"] not in done]
    print(f"{name}: {len(todo)} clips to label ({len(done)} done) with {model.name}", flush=True)
    started, audio_s, decode_s = time.perf_counter(), 0.0, 0.0
    with open(out, "a", encoding="utf-8") as handle:
        for i, clip in enumerate(todo, 1):
            samples = audio.load(folder / clip["audio"])
            tick = time.perf_counter()
            result = whistle.transcribe(samples, language=clip["language"])
            seconds = time.perf_counter() - tick
            audio_s, decode_s = audio_s + clip["duration"], decode_s + seconds
            handle.write(json.dumps({"id": clip["id"], "teacher": result["text"], "language": result["language"],
                                     "decode_s": round(seconds, 4)}, ensure_ascii=False) + "\n")
            if i % every == 0 or i == len(todo):
                handle.flush()
                print(f"  {i}/{len(todo)}  {time.perf_counter() - started:6.0f} s  "
                      f"{audio_s / max(decode_s, 1e-9):5.1f}x real time", flush=True)
    return {"labelled": len(done) + len(todo), "file": str(out)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.label", description=__doc__.split("\n\n")[0])
    parser.add_argument("source")
    parser.add_argument("--model", type=Path, default=Path("models/shruti/shruti.cact"))
    args = parser.parse_args(argv)
    print(json.dumps(label(args.source, args.model.resolve())))


if __name__ == "__main__":
    main()
