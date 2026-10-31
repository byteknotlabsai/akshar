"""Write a Whistle checkpoint to a .cact file with a chosen CQ width per weight matrix.

needle's exporter (needle/model/export.py) writes every matrix at 4 bits and refuses any other width, although the
.cact format and the engine accept CQ widths 1 to 4: Cactus's own base whistle.cact is mostly 2-bit. This module
reuses needle's writer unchanged and swaps in one function that picks each matrix's width from a recipe.

A recipe is a function (tensor name, default bits) -> bits. With `uniform(4)` the output is byte-identical to
`needle whistle build` (tests/test_export.py).
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .checkpoint import Checkpoint

CQ_GROUP = 128
CQ_WIDTHS = (1, 2, 3, 4)


def uniform(bits: int):
    """Every CQ matrix at the same width."""
    return lambda name, default: bits


def cactus_mixed(name: str, default: int) -> int:
    """Cactus's recipe for its own 2-bit whistle.cact: the token embedding and the mHC routing matrices
    (mhc_phi_*) at 4 bits, every other matrix at 2 bits. Read from the published file's tensor directory."""
    return 4 if name == "embedding" or name.rsplit(".", 1)[-1].startswith("mhc_phi") else 2


def cq_pack(w: np.ndarray, bits: int, group: int = CQ_GROUP):
    """needle's _cq_pack without its 4-bit-only guard: indices LSB-first, then per-group FP16 norms."""
    from needle.model import export as E
    from needle.model.quantize import _cq_codebook_np, _cq_hadamard_np

    if bits not in CQ_WIDTHS:
        raise ValueError(f"CQ width must be one of {CQ_WIDTHS}, got {bits}")
    cb = _cq_codebook_np(bits, group)
    H = _cq_hadamard_np(group)
    out, D = w.shape
    pad = (-D) % group
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    in_pad = wp.shape[1]
    g = wp.reshape(out, in_pad // group, group).astype(np.float32)
    rot = g @ H
    norm = np.sqrt((rot ** 2).sum(-1, keepdims=True))
    unit = rot / np.maximum(norm, 1e-12)
    idx = E._nearest_idx(unit, cb).reshape(out, in_pad)
    return E._pack_lsb(idx, bits), norm[:, :, 0].astype(np.float16)


@contextmanager
def _recipe(recipe):
    from needle.model import export as E

    original = E._q
    chosen = {}

    def q(name, mat, bits, group):
        width = recipe(name, bits)
        chosen[name] = width
        packed, norms = cq_pack(np.asarray(mat, np.float32), width, group)
        return E._Tensor(name, E.CQ, mat.shape, packed.tobytes() + norms.tobytes(), group, width)

    E._q = q
    try:
        yield chosen
    finally:
        E._q = original


def write(checkpoint: Checkpoint, out, recipe=uniform(4), layers: int | None = None) -> dict:
    """Write `checkpoint` to `out` (.cact). `layers` keeps only the first N decoder layers (needle's depth rungs)."""
    from needle.model.whistle_finetune import build, rung

    params, cfg = rung(checkpoint.params, checkpoint.cfg, layers)
    with _recipe(recipe) as chosen:
        info = build(params, cfg, checkpoint.tokenizer(), str(out))
    widths = {}
    for width in chosen.values():
        widths[width] = widths.get(width, 0) + 1
    return {**info, "cq_widths": dict(sorted(widths.items())), "dec_layers": cfg.dec_layers}
