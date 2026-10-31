"""Training-time weight quantisation that picks the same CQ codes as the .cact exporter.

needle's cq_quantize rotates each 128-weight group with a Hadamard matrix using a plain float32 matmul. On an
Ampere GPU, JAX runs that matmul in TF32 by default, so training then quantises to different codes than the
exporter (numpy, full float32) writes: about 680 codes per million at 4 bits and 150 at 2 bits on Shruti, and
68,117 group norms. With the rotation at HIGHEST precision the two disagree on fewer than 1 code per million,
which is float rounding. These functions are needle's, with that precision fixed and a width per matrix.

Widths come from a recipe, as in akshar.model.export: (name, default bits) -> bits. Training leaf names differ from
.cact tensor names, so `train_recipe` maps a .cact recipe onto training leaves.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST
GROUP = 128


def cq_quantize(w, bits, group=GROUP):
    """needle.model.quantize.cq_quantize with both Hadamard rotations at HIGHEST precision."""
    from needle.model.quantize import _cq_codebook_np, _cq_hadamard_np, _cq_nearest

    cb = jnp.asarray(_cq_codebook_np(bits, group))
    D = w.shape[-1]
    pad = (-D) % group
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    groups = wp.reshape(*wp.shape[:-1], -1, group).astype(jnp.float32)
    H = jnp.asarray(_cq_hadamard_np(group))
    rot = jnp.matmul(groups, H, precision=HIGHEST)
    norm = jnp.sqrt(jnp.sum(rot ** 2, axis=-1, keepdims=True))
    unit = rot / jnp.maximum(norm, 1e-12)
    norm = norm.astype(jnp.float16).astype(jnp.float32)
    deq = jnp.matmul(_cq_nearest(unit, cb) * norm, H, precision=HIGHEST)
    deq = deq.reshape(wp.shape).astype(w.dtype)
    return deq[..., :D] if pad else deq


def cq_ste(w, bits, group=GROUP):
    """Quantised in the forward pass, straight-through in the backward pass."""
    return w + jax.lax.stop_gradient(cq_quantize(w, bits, group) - w)


def leaf_widths(params, recipe) -> dict[str, int]:
    """CQ width of every quantised training leaf, keyed by its slash-separated path."""
    from needle.model.quantize import _is_quant_leaf, leaf_name

    widths = {}

    def visit(path, leaf):
        if _is_quant_leaf(path, leaf):
            widths[leaf_name(path)] = recipe(leaf_name(path), 4)
        return leaf

    jax.tree_util.tree_map_with_path(visit, params)
    return widths


def cq_ste_params(params, widths: dict[str, int], group=GROUP):
    """Fake-quantise every CQ leaf at its width, along the axis the exporter groups along."""
    from needle.model.quantize import _is_quant_leaf, _reduces_second_last, leaf_name

    def q(path, leaf):
        if not _is_quant_leaf(path, leaf):
            return leaf
        bits = widths[leaf_name(path)]
        if _reduces_second_last(path):
            return jnp.swapaxes(cq_ste(jnp.swapaxes(leaf, -1, -2), bits, group), -1, -2)
        return cq_ste(leaf, bits, group)

    return jax.tree_util.tree_map_with_path(q, params)


def train_recipe(export_recipe):
    """Turn a .cact recipe (keyed by exported tensor names) into one keyed by training leaf paths.

    Only two exported names matter to the recipes we use: "embedding" (the token embedding, training leaf
    "embedding/embedding") and the mhc_phi_* routing matrices (same key in training). Engram tables are training
    leaves named ".../embedding" too, but they are exported as "engram{i}.tables".
    """
    def recipe(leaf: str, default: int) -> int:
        if leaf == "embedding/embedding":
            return export_recipe("embedding", default)
        last = leaf.rsplit("/", 1)[-1]
        if last.startswith("mhc_phi"):
            return export_recipe(f"encoder.{last}" if leaf.startswith("encoder/") else last, default)
        return export_recipe(leaf, default)

    return recipe
