"""Load and save full-precision Whistle checkpoints (.safetensors), keeping the tokenizer they were trained with.

A checkpoint's metadata holds the model config and a `run` record. Shruti's run record lists the 714 Devanagari
pieces added to Whistle's tokenizer; without them the vocabulary and the embedding rows would not line up.
"""
from __future__ import annotations

from dataclasses import dataclass, replace



@dataclass
class Checkpoint:
    params: dict
    cfg: object  # needle.model.whistle.WhistleConfig
    run: dict
    step: int | None = None

    @property
    def pieces(self) -> tuple[str, ...]:
        return tuple(self.run.get("pieces") or ())

    def tokenizer(self):
        from needle.model.whistle_finetune import WhistleTokenizer

        tokenizer = WhistleTokenizer(self.pieces, self.cfg.languages)
        if tokenizer.text != self.cfg.text:
            raise ValueError(f"tokenizer has {tokenizer.text} pieces but the model expects {self.cfg.text}")
        return tokenizer


def load(path) -> Checkpoint:
    from needle.model.checkpoints import read_checkpoint
    from needle.model.whistle import WhistleConfig

    raw = read_checkpoint(str(path))
    if "encoder" not in raw["params"] or "stem" not in raw["params"]:
        raise ValueError(f"{path} is not a Whistle checkpoint")
    return Checkpoint(raw["params"], WhistleConfig.from_saved(raw["config"]), raw["run"] or {}, raw["step"])


def save(path, checkpoint: Checkpoint) -> None:
    import dataclasses

    import numpy as np
    from needle.model.checkpoints import write_checkpoint

    params = _to_numpy(checkpoint.params, np)
    config = dataclasses.asdict(checkpoint.cfg)
    write_checkpoint(str(path), {"params": params, "config": config, "run": checkpoint.run, "step": checkpoint.step})


def _to_numpy(tree, np):
    if isinstance(tree, dict):
        return {k: _to_numpy(v, np) for k, v in tree.items()}
    return np.asarray(tree, np.float32)


def with_params(checkpoint: Checkpoint, params, step=None) -> Checkpoint:
    return replace(checkpoint, params=params, step=step)
