"""Full-weight fine-tuning of a Whistle checkpoint with quantisation-aware training (roadmap Phase 1).

Every weight trains, not adapters. Each CQ matrix is fake-quantised at the width it will be exported with, by
akshar.model.quant, so the .cact runs the numerics the model trained on. Activations and the KV cache are
fake-quantised to 8 bits, as in needle's trainer and the engine.

    python -m akshar.train.finetune --checkpoint models/shruti/shruti.safetensors \
        --train ~/akshar-data/train/fleurs_hi_train/manifest.jsonl --recipe w4 --steps 100 --out runs/train/demo

Writes <out>/model.safetensors (full precision, same metadata layout as Shruti's), <out>/model-<recipe>.cact and
<out>/train.json. This Phase 1 version loads the whole manifest into memory and pads every clip to 30 s, as needle's
trainer does. That suits a few hours of audio; Phase 3 needs a streaming loader.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..model import checkpoint, export

RECIPES = {"w4": export.uniform(4), "w2": export.uniform(2), "mixed2": export.cactus_mixed}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.train.finetune", description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="full-precision .safetensors to start from")
    parser.add_argument("--train", required=True, help="manifest JSONL: audio, text, language")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="w4", help="CQ widths to train and export at")
    parser.add_argument("--steps", type=int, default=100, help="optimiser updates")
    parser.add_argument("--batch", type=int, default=4, help="clips per forward pass")
    parser.add_argument("--accum", type=int, default=4, help="forward passes per optimiser update")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from needle.model.quantize import configure_deploy
    from needle.model.whistle import Whistle
    from needle.model.whistle_finetune import _arrays, read_clips

    from ..model.quant import cq_ste_params, leaf_widths, train_recipe

    ck = checkpoint.load(args.checkpoint)
    cfg, tokenizer = ck.cfg, ck.tokenizer()
    clips, skipped = read_clips(args.train)
    items = [(samples, [cfg.bos, cfg.tag(lang), *tokenizer.encode(text), cfg.eos], text)
             for samples, text, lang in clips]
    too_long = sum(len(ids) > cfg.max_tokens + 1 for _, ids, _ in items)
    items = [item for item in items if len(item[1]) <= cfg.max_tokens + 1]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(items))
    n_val = max(args.batch, int(len(items) * args.val_fraction))
    val, train = [items[i] for i in order[:n_val]], [items[i] for i in order[n_val:]]
    hours = sum(len(s) for s, _, _ in train) / 16000 / 3600
    print(f"data      {len(train)} train clips ({hours:.2f} h), {len(val)} validation; "
          f"skipped {skipped} unreadable or over 30 s, {too_long} over {cfg.max_tokens} tokens", flush=True)

    configure_deploy(act_bits=8, kv_bits=8)
    model = Whistle(cfg)
    params = jax.device_put(jax.tree.map(lambda a: jnp.asarray(a, jnp.float32), ck.params))
    widths = leaf_widths(params, train_recipe(RECIPES[args.recipe]))
    print(f"quant     {args.recipe}: {sum(b == 4 for b in widths.values())} leaves at 4 bits, "
          f"{sum(b == 2 for b in widths.values())} at 2 bits; activations and KV at 8 bits", flush=True)

    def loss_fn(params, wav, lengths, tokens):
        logits = model.apply({"params": cq_ste_params(params, widths)}, wav, lengths, tokens[:, :-1], quant=True)
        targets = tokens[:, 1:]
        mask = targets != cfg.pad
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    warmup = max(1, args.steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, args.steps, end_value=args.lr / 10)
    optimizer = optax.MultiSteps(optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=0.0)),
                                 every_k_schedule=args.accum)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, wav, lengths, tokens):
        loss, grads = jax.value_and_grad(loss_fn)(params, wav, lengths, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    eval_loss = jax.jit(loss_fn)

    def validation_loss(params):
        losses = [float(eval_loss(params, *_arrays(val, range(i, i + args.batch), cfg)))
                  for i in range(0, len(val) - args.batch + 1, args.batch)]
        return float(np.mean(losses))

    def batches():
        while True:
            perm = rng.permutation(len(train))
            for i in range(0, len(perm) - args.batch + 1, args.batch):
                idx = perm[i:i + args.batch]
                yield _arrays(train, idx, cfg), sum(len(train[j][0]) for j in idx) / 16000

    start_val = validation_loss(params)
    print(f"val loss  {start_val:.4f} before training", flush=True)
    log, window, audio_s, started = [], [], 0.0, None
    stream = batches()
    for micro in range(args.steps * args.accum):
        arrays, seconds = next(stream)
        params, opt_state, loss = train_step(params, opt_state, *arrays)
        if micro == 0:
            jax.block_until_ready(loss)
            started = time.perf_counter()  # the first call compiles; time the rest
            continue
        window.append(float(loss))
        audio_s += seconds
        step = (micro + 1) // args.accum
        if (micro + 1) % args.accum == 0 and step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            entry = {"step": step, "loss": round(float(np.mean(window)), 4), "elapsed_s": round(elapsed, 1),
                     "audio_per_second": round(audio_s / elapsed, 1)}
            log.append(entry)
            print(f"step      {step}/{args.steps}  loss {entry['loss']:.4f}  {elapsed:6.0f} s  "
                  f"{entry['audio_per_second']:.0f} s of audio per s", flush=True)
            window = []
    end_val = validation_loss(params)
    print(f"val loss  {end_val:.4f} after training", flush=True)

    run = {**ck.run, "mode": "full", "akshar": {
        "phase": 1, "started_from": str(args.checkpoint), "train": str(args.train), "recipe": args.recipe,
        "steps": args.steps, "batch": args.batch, "accum": args.accum, "lr": args.lr, "seed": args.seed}}
    trained = checkpoint.Checkpoint(jax.tree.map(np.asarray, params), cfg, run, step=args.steps)
    checkpoint.save(args.out / "model.safetensors", trained)
    info = export.write(trained, args.out / f"model-{args.recipe}.cact", RECIPES[args.recipe])
    peak = jax.devices()[0].memory_stats() or {}
    summary = {
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), **run["akshar"],
        "train_clips": len(train), "train_hours": round(hours, 3), "val_clips": len(val),
        "val_loss_before": round(start_val, 4), "val_loss_after": round(end_val, 4), "log": log,
        "gpu_peak_bytes": peak.get("peak_bytes_in_use"), "export": {k: v for k, v in info.items() if k != "path"},
    }
    (args.out / "train.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"wrote     {args.out / 'model.safetensors'} and {info['path']} ({info['bytes'] / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
