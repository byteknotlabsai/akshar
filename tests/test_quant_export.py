"""akshar's quantiser and exporter against needle's own code (needs models/shruti/shruti.safetensors for some tests)."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

SHRUTI = Path(__file__).resolve().parents[1] / "models" / "shruti" / "shruti.safetensors"
needs_shruti = pytest.mark.skipif(not SHRUTI.exists(), reason="models/shruti/shruti.safetensors not downloaded")


@pytest.mark.parametrize("bits", [2, 4])
def test_quantiser_matches_needle_on_cpu(bits):
    import jax.numpy as jnp
    from needle.model.quantize import cq_quantize as needle_cq

    from akshar.model.quant import cq_quantize

    w = jnp.asarray(np.random.default_rng(bits).standard_normal((64, 300)).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(cq_quantize(w, bits)), np.asarray(needle_cq(w, bits)))


@pytest.mark.parametrize("bits", [2, 4])
def test_exporter_codes_round_trip(bits):
    from needle.model import export as E

    from akshar.model.export import cq_pack

    w = np.random.default_rng(10 + bits).standard_normal((32, 256)).astype(np.float32)
    packed, norms = cq_pack(w, bits)
    restored = E._cq_unpack(packed.reshape(32, -1), norms, 32, 256, bits, 128)
    from akshar.model.quant import cq_quantize

    np.testing.assert_allclose(restored, np.asarray(cq_quantize(w, bits)), rtol=0, atol=1e-6)


@needs_shruti
def test_mixed_recipe_gives_trainer_and_exporter_the_same_4_bit_weights():
    from needle.model import export as E

    from akshar.model import checkpoint
    from akshar.model.export import cactus_mixed
    from akshar.model.quant import leaf_widths, train_recipe

    ck = checkpoint.load(SHRUTI)
    sizes = {}

    def leaf_size(params, name):
        node = params
        for part in name.split("/"):
            node = node[part]
        return int(np.prod(np.shape(node)))

    widths = leaf_widths(ck.params, train_recipe(cactus_mixed))
    trained_4bit = sum(leaf_size(ck.params, n) for n, b in widths.items() if b == 4)

    original = E._cq_pack
    try:
        E._cq_pack = lambda w, bits, group: (sizes.setdefault(len(sizes), (w.size, w.shape)) and None) or (
            np.zeros((w.shape[0], 1), np.uint8), np.zeros((w.shape[0], 1), np.float16))
        names = []
        original_q = E._q
        E._q = lambda name, mat, bits, group: names.append((name, np.asarray(mat).size)) or original_q(
            name, mat, bits, group)
        E._tensors(ck.params, ck.cfg.needle(), 4, 128, audio=ck.cfg)
    finally:
        E._cq_pack, E._q = original, original_q
    exported_4bit = sum(size for name, size in names if cactus_mixed(name, 4) == 4)
    assert trained_4bit == exported_4bit > 0
    assert sum(leaf_size(ck.params, n) for n in widths) == sum(size for _, size in names)


@needs_shruti
def test_uniform_4_bit_export_equals_needle_build(tmp_path):
    from needle.model.whistle_finetune import build

    from akshar.model import checkpoint, export

    ck = checkpoint.load(SHRUTI)
    ours = export.write(ck, tmp_path / "ours.cact", export.uniform(4))
    build(ck.params, ck.cfg, ck.tokenizer(), str(tmp_path / "needle.cact"))
    digest = [hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in ("ours.cact", "needle.cact")]
    assert ours["cq_widths"] == {4: 152} and digest[0] == digest[1]
