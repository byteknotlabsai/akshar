"""Track B's starting point: Whistle's own weights, given Shruti's tokenizer (D9: akshar is trained from Whistle).

    python -m akshar.model.track_b --out models/whistle-hi/whistle-hi-init.safetensors

Shruti added 714 Devanagari pieces and a `hi` language tag to Whistle's tokenizer. This builds the same vocabulary on
Whistle's full-precision weights with needle's own code, as Shruti's report describes: each new piece's embedding row
starts as the mean of the byte-piece rows Whistle's tokenizer would have used for it, and the `hi` tag as the mean of
the existing tag rows. Every other weight is Whistle's. Because the tokenizer is identical to Shruti's, Shruti can
teach it token by token (akshar.train.distill). Whistle's checkpoint is downloaded from Hugging Face if missing.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from . import checkpoint


def build(shruti: Path, out: Path, base: str) -> dict:
    import jax
    import numpy as np
    from needle.model.whistle_finetune import WhistleTokenizer, initial_rows, load_base, tuned_params

    ref = checkpoint.load(shruti)
    params, cfg = load_base(base)
    differ = {k: (v, getattr(ref.cfg, k)) for k, v in dataclasses.asdict(cfg).items() if getattr(ref.cfg, k) != v}
    allowed = {"text", "vocab", "vocab_size", "languages", "n_languages"}
    if set(differ) - allowed:
        raise SystemExit(f"Whistle and Shruti differ beyond the vocabulary: {sorted(set(differ) - allowed)}")
    tokenizer = WhistleTokenizer(ref.pieces, ref.cfg.languages)
    rows = initial_rows(params["embedding"]["embedding"], cfg, tokenizer)
    student = tuned_params(params, cfg, tokenizer, None, None, rows)
    shapes = lambda tree: {"/".join(str(getattr(k, "key", k)) for k in p): np.shape(v)
                           for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}
    mine, theirs = shapes(student), shapes(ref.params)
    if mine != theirs:
        bad = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        raise SystemExit(f"shapes differ from Shruti's at {bad[:5]}")
    run = {"mode": "full", "pieces": list(ref.pieces), "akshar": {
        "track": "B", "init": "whistle", "base": base, "tokenizer_from": str(shruti), "new_rows": int(len(rows))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.save(out, checkpoint.Checkpoint(student, ref.cfg, run, step=0))
    return {"out": str(out), "differences": {k: [str(a), str(b)] for k, (a, b) in differ.items()},
            "new_rows": int(len(rows)), "leaves": len(mine)}


def main(argv=None) -> None:
    from needle.model.whistle_finetune import DEFAULT_BASE

    parser = argparse.ArgumentParser(prog="python -m akshar.model.track_b", description=__doc__.split("\n\n")[0])
    parser.add_argument("--shruti", type=Path, default=Path("models/shruti/shruti.safetensors"))
    parser.add_argument("--base", default=DEFAULT_BASE, help="Whistle's full-precision checkpoint")
    parser.add_argument("--out", type=Path, default=Path("models/whistle-hi/whistle-hi-init.safetensors"))
    args = parser.parse_args(argv)
    print(json.dumps(build(args.shruti, args.out, args.base)))


if __name__ == "__main__":
    main()
