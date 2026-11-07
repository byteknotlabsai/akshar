"""Export a run's latest saved state (state.msgpack) to a checkpoint and a .cact, without finishing the run.

    python -m akshar.train.export_state runs/train/k1 --student models/whistle-hi/whistle-hi-init.safetensors --recipe w4

akshar.train.distill saves its weights and optimiser state at every validation step and exports only when the run
completes. This writes <run>/model-latest.safetensors and <run>/model-latest-<recipe>.cact from the last save, so a
run stopped by a deadline still yields a model. The student checkpoint supplies the tensor layout, config and
tokenizer; the run is untouched and can still resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..model import checkpoint, export
from .distill import RECIPES


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.train.export_state", description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--student", type=Path, required=True, help="the checkpoint the run started from")
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    parser.add_argument("--layers", type=int)
    args = parser.parse_args(argv)

    import jax
    import numpy as np
    import optax
    from flax import serialization
    from needle.model.whistle_finetune import rung

    start = checkpoint.load(args.student)
    params, cfg = rung(start.params, start.cfg, args.layers)
    params = jax.tree.map(lambda a: np.asarray(a, np.float32), params)
    raw = serialization.msgpack_restore((args.run / "state.msgpack").read_bytes())
    restored = serialization.from_state_dict(params, raw["params"])
    micro = int(raw.get("micro", 0))
    run = {**start.run, "akshar": {**start.run.get("akshar", {}), "exported_from_state": str(args.run), "micro": micro}}
    ck = checkpoint.Checkpoint(jax.tree.map(np.asarray, restored), cfg, run, step=micro)
    checkpoint.save(args.run / "model-latest.safetensors", ck)
    info = export.write(ck, args.run / f"model-latest-{args.recipe}.cact", RECIPES[args.recipe])
    print(json.dumps({"micro_steps": micro, "cact": str(info["path"]), "bytes": info["bytes"]}))


if __name__ == "__main__":
    main()
