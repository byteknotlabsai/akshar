"""Quantisation-aware training of a Whistle student against a frozen teacher (roadmap Phase 2, Track A).

The student starts from a full-precision checkpoint and trains every weight, fake-quantised at its recipe's widths.
Its targets are the teacher's own transcripts (akshar.data.label); silence and noise clips (akshar.data.nonspeech)
have empty targets, so the model learns to write nothing on them. The loss on each token is

    (1 - alpha) * cross-entropy(teacher's transcript)  +  alpha * T^2 * KL(teacher || student at temperature T)

so alpha 0 is plain quantisation-aware training on the teacher's transcripts, and alpha > 0 adds distillation from the
teacher's full output distribution, which the roadmap's exit check compares.

    python -m akshar.train.distill --sources librispeech_100 rural_women_hi --recipe mixed2 --alpha 0.5 \
        --steps 2000 --out runs/train/a2-kd

With --pool, more clips stream in during the run: akshar.train.feed keeps a few Drive shards at a time in that folder,
and the run takes --pool-share of its batches from them (see Pool). --val-only sources give validation clips only, so a
source can move to the stream and keep its place in the validation loss.

Writes <out>/model.safetensors, <out>/model-<recipe>.cact, <out>/state.msgpack (for --resume) and <out>/train.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..eval.paths import DATA_ROOT
from ..model import checkpoint, export

RECIPES = {"w4": export.uniform(4), "w2": export.uniform(2), "mixed2": export.cactus_mixed,
           "mixed3": export.cactus_mixed_at(3), "mixed2-dec4": export.cactus_mixed_plus("decoder"),
           "mixed2-enc4": export.cactus_mixed_plus("encoder"), "mixed2-engram4": export.cactus_mixed_plus("engram"),
           "mixed2-dec3": export.cactus_mixed_plus("decoder", 3), "mixed2-cross4": export.cactus_mixed_plus("cross"),
           "mixed2-enc3": export.cactus_mixed_plus("encoder", 3),
           "mixed2-encdec3": export.cactus_mixed_with({"encoder": 3, "decoder": 3}),
           "mixed2-enc4dec3": export.cactus_mixed_with({"encoder": 4, "decoder": 3})}


def _is_validation(clip_id: str, fraction: float) -> bool:
    return int(hashlib.sha256(clip_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def read_folder(folder: Path, labels: str, tokenizer, cfg):
    """(clip id, Clip) for every clip in one folder that has a target and fits the model, and the folder's counts."""
    from .loader import Clip

    with open(folder / f"labels-{labels}.jsonl", encoding="utf-8") as handle:
        teacher = {row["id"]: row for row in map(json.loads, handle)}
    clips, kept, empty, unlabelled, too_long, other_language = [], 0, 0, 0, 0, 0
    with open(folder / "manifest.jsonl", encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            label = teacher.get(row["id"])
            if row["duration"] > 30.0:  # the model's window; longer clips cannot be trained on whole
                too_long += 1
                continue
            if label is None:
                unlabelled += 1
                continue
            text, language = label["teacher"].strip(), label["language"] or row["language"]
            if language not in cfg.languages:
                other_language += 1
                continue
            # An empty target is kept: on silence and noise the model must learn to write nothing and stop (D22).
            ids = (cfg.bos, cfg.tag(language), *tokenizer.encode(text), cfg.eos)
            if len(ids) > cfg.max_tokens + 1:
                continue
            clips.append((row["id"], Clip(folder / row["audio"], row["duration"], ids)))
            kept += 1
            empty += not text
    return clips, {"clips": kept, "empty_target": empty, "unlabelled": unlabelled, "over_30s": too_long,
                   "other_language": other_language}


def load_clips(sources: list[str], labels: str, tokenizer, cfg, val_fraction: float, val_only: list[str] = (),
               weights: dict[str, int] | None = None):
    train, val, stats = [], [], {}
    for name in [*sources, *val_only]:
        clips, stats[name] = read_folder(DATA_ROOT / "train" / name, labels, tokenizer, cfg)
        for clip_id, clip in clips:
            if _is_validation(clip_id, val_fraction):
                val.append(clip)
            elif name not in val_only:
                train.extend([clip] * (weights or {}).get(name, 1))
        if name in val_only:
            stats[name]["validation_only"] = True
    return train, val, stats


class Pool:
    """Clips streamed through a folder by akshar.train.feed while the run trains.

    <folder>/pool.json lists the shard folders that are ready, each with manifest.jsonl, labels-<labels>.jsonl and
    its clips. `refresh` reads it; when the list has changed, it builds new batches over those folders, with the fixed
    data's bucket shapes so that nothing recompiles, lets go of the old batches, and writes <folder>/inuse.json. The
    feeder deletes a folder only once inuse.json, written after the folder left pool.json, no longer lists it.
    Validation clips (by id, as in load_clips) are never trained on, so a source can move from disk to the stream.
    """

    def __init__(self, folder: Path, labels: str, tokenizer, cfg, budget: float, tokens: list[int],
                 val_fraction: float, used: dict[str, float] | None = None):
        self.dir, self.labels, self.tokenizer, self.cfg = folder, labels, tokenizer, cfg
        self.budget, self.tokens, self.val_fraction = budget, tokens, val_fraction
        self.listed: list[str] = []
        self.clips: dict[str, list] = {}
        self.batches = None
        self.used: dict[str, float] = dict(used or {})  # every shard folder this run has trained on: hours

    def refresh(self, seed: int) -> str | None:
        from .loader import Batches

        try:
            listed = list(json.loads((self.dir / "pool.json").read_text(encoding="utf-8"))["folders"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if listed == self.listed:
            return None
        for name in listed:
            if name not in self.clips:
                try:
                    clips, _ = read_folder(self.dir / name, self.labels, self.tokenizer, self.cfg)
                except OSError as error:
                    print(f"pool      cannot read {name}: {error}", flush=True)
                    clips = []
                self.clips[name] = [c for clip_id, c in clips if not _is_validation(clip_id, self.val_fraction)]
                self.used[name] = round(sum(c.seconds for c in self.clips[name]) / 3600, 3)
        for name in [n for n in self.clips if n not in listed]:
            del self.clips[name]
        clips = [c for name in listed for c in self.clips[name]]
        old = self.batches
        try:
            self.batches = Batches(clips, self.budget, self.cfg.max_tokens, self.cfg.pad, seed=seed,
                                   tokens=self.tokens)
        except ValueError:
            self.batches = None
        if old is not None:
            old.close()
        self.listed = listed
        note = self.dir / "inuse.json.tmp"
        note.write_text(json.dumps({"folders": listed if self.batches else [], "time": time.time()}) + "\n")
        note.replace(self.dir / "inuse.json")
        hours = sum(c.seconds for c in clips) / 3600
        if self.batches is None:
            return f"pool      {len(listed)} folders, {hours:.1f} h: too few clips for a batch yet"
        return (f"pool      {len(listed)} folders, {hours:.1f} h ({self.batches.dropped} clips too long for the "
                f"fixed shapes); {len(self.used)} folders so far, {sum(self.used.values()):.0f} h")

    def by_source(self) -> dict[str, float]:
        """Hours streamed per source; shard folders are named <source>__<shard>."""
        totals: dict[str, float] = {}
        for name, hours in self.used.items():
            source = name.split("__")[0]
            totals[source] = round(totals.get(source, 0.0) + hours, 2)
        return totals


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.train.distill", description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", nargs="+", required=True, help="training sets under $AKSHAR_DATA/train")
    parser.add_argument("--val-only", nargs="+", default=[], metavar="SOURCE",
                        help="sets under $AKSHAR_DATA/train that give only validation clips (their audio for the rest "
                             "may be gone)")
    parser.add_argument("--weight", nargs="+", default=[], metavar="SOURCE=N",
                        help="take a source's training clips N times as often (its validation clips stay single)")
    parser.add_argument("--pool", type=Path, help="folder that akshar.train.feed streams shard folders through")
    parser.add_argument("--pool-share", type=float, default=0.6, help="share of forward passes taken from the pool")
    parser.add_argument("--refresh", type=int, default=100, help="read the pool's list every N updates")
    parser.add_argument("--labels", default="shruti", help="teacher labels file: labels-<this>.jsonl")
    parser.add_argument("--student", default="models/shruti/shruti.safetensors")
    parser.add_argument("--teacher", default="models/shruti/shruti.safetensors")
    parser.add_argument("--teacher-quant", choices=["w4", "none"], default="w4",
                        help="run the teacher as deployed (4-bit weights, 8-bit activations) or in full precision")
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="mixed2")
    parser.add_argument("--layers", type=int, help="keep only the first N decoder layers (needle's depth rungs)")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of the distillation term; 0 = none")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000, help="optimiser updates")
    parser.add_argument("--budget", type=float, default=120.0, help="seconds of padded audio per forward pass")
    parser.add_argument("--accum", type=int, default=4, help="forward passes per optimiser update")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=250, help="validation loss every N updates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="continue from <out>/state.msgpack")
    parser.add_argument("--track", choices=["A", "B"],
                        help="A: weights descend from Shruti's (research-only, D9); B: from Whistle's. "
                             "Default: the student's own record, else A")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import serialization
    from needle.model.quantize import configure_deploy
    from needle.model.whistle import Whistle
    from needle.model.whistle_finetune import rung

    from ..model.quant import cq_ste_params, leaf_widths, train_recipe
    from .loader import Batches, plan, _assemble

    student_ck, teacher_ck = checkpoint.load(args.student), checkpoint.load(args.teacher)
    if student_ck.pieces != teacher_ck.pieces or student_ck.cfg.languages != teacher_ck.cfg.languages:
        raise SystemExit("student and teacher must share a tokenizer for token-level distillation")
    tokenizer, teacher_cfg = teacher_ck.tokenizer(), teacher_ck.cfg
    student_params, cfg = rung(student_ck.params, student_ck.cfg, args.layers)
    weights = {name: int(n) for name, n in (w.split("=") for w in args.weight)}
    train, val, stats = load_clips(args.sources, args.labels, tokenizer, cfg, args.val_fraction, args.val_only, weights)
    print(f"data      {len(train)} train clips ({sum(c.seconds for c in train) / 3600:.1f} h), {len(val)} validation; "
          f"{json.dumps(stats)}", flush=True)

    configure_deploy(act_bits=8, kv_bits=8)
    student, teacher = Whistle(cfg), Whistle(teacher_cfg)
    params = jax.device_put(jax.tree.map(lambda a: jnp.asarray(a, jnp.float32), student_params))
    with jax.default_device(jax.devices("cpu")[0]):  # quantise the frozen teacher once, on the CPU, so only
        teacher_params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float32), teacher_ck.params)  # one copy
        if args.teacher_quant == "w4":  # ever reaches the GPU
            teacher_params = cq_ste_params(teacher_params, leaf_widths(teacher_params, lambda name, default: 4))
    teacher_params = jax.device_put(teacher_params, jax.devices()[0])
    widths = leaf_widths(params, train_recipe(RECIPES[args.recipe]))
    alpha, temp = args.alpha, args.temperature

    # The teacher's weights are an argument, never captured: jit bakes captured arrays into every compiled
    # program as constants, which put a 222 MB copy of the teacher on the GPU for each batch shape.
    def losses(params, teacher_params, wav, lengths, tokens):
        logits = student.apply({"params": cq_ste_params(params, widths)}, wav, lengths, tokens[:, :-1], quant=True)
        targets = tokens[:, 1:]
        mask = (targets != cfg.pad).astype(jnp.float32)
        count = jnp.maximum(mask.sum(), 1.0)
        ce = (optax.softmax_cross_entropy_with_integer_labels(logits, targets) * mask).sum() / count
        if alpha == 0:
            return ce, (ce, jnp.zeros(()))
        t_logits = jax.lax.stop_gradient(teacher.apply({"params": teacher_params}, wav, lengths, tokens[:, :-1],
                                                       quant=args.teacher_quant == "w4"))
        t_logp = jax.nn.log_softmax(t_logits / temp, axis=-1)
        s_logp = jax.nn.log_softmax(logits / temp, axis=-1)
        kl = ((jnp.exp(t_logp) * (t_logp - s_logp)).sum(-1) * mask).sum() / count
        return (1 - alpha) * ce + alpha * temp ** 2 * kl, (ce, kl)

    warmup = max(1, args.steps // 20)
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, args.steps, end_value=args.lr / 20)
    optimizer = optax.MultiSteps(optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=0.0)),
                                 every_k_schedule=args.accum)
    opt_state = optimizer.init(params)
    state_path, micro_done, log, pool_used = args.out / "state.msgpack", 0, [], {}
    if args.resume and state_path.exists():
        # Restore the arrays into their templates, and take the validation log as saved: flax stores a list as a dict
        # keyed "0", "1", ..., and an empty template list would not match it.
        raw = serialization.msgpack_restore(state_path.read_bytes())
        params = serialization.from_state_dict(params, raw["params"])
        opt_state = serialization.from_state_dict(opt_state, raw["opt_state"])
        micro_done = int(raw["micro"])
        saved_log = raw.get("log") or {}
        log = [saved_log[k] for k in sorted(saved_log, key=int)] if isinstance(saved_log, dict) else list(saved_log)
        pool_used = {k: float(v) for k, v in (raw.get("pool_used") or {}).items()}
        print(f"resumed   at update {micro_done // args.accum}", flush=True)

    @jax.jit
    def train_step(params, opt_state, teacher_params, wav, lengths, tokens):
        (loss, parts), grads = jax.value_and_grad(losses, has_aux=True)(params, teacher_params, wav, lengths, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, parts

    eval_losses = jax.jit(lambda p, tp, w, l, t: losses(p, tp, w, l, t)[1])
    val_buckets, val_groups = plan(val, args.budget, cfg.max_tokens)
    val_batches = [(b, g[i:i + bucket.batch]) for b, (bucket, g) in enumerate(zip(val_buckets, val_groups))
                   for i in range(0, len(g) - bucket.batch + 1, bucket.batch)]
    val_arrays = [(b, _assemble(val_buckets[b], clips, cfg.pad)) for b, clips in val_batches]

    def validate(params):
        parts = [tuple(float(x) for x in eval_losses(params, teacher_params, *arrays)) for _, arrays in val_arrays]
        return {"ce": round(float(np.mean([p[0] for p in parts])), 4),
                "kl": round(float(np.mean([p[1] for p in parts])), 4)}

    def save_state(micro):
        state = {"params": params, "opt_state": opt_state, "micro": micro, "log": log}
        if pool is not None and pool.used:
            state["pool_used"] = pool.used
        state_path.write_bytes(serialization.to_bytes(state))

    batches = Batches(train, args.budget, cfg.max_tokens, cfg.pad, seed=args.seed + micro_done)
    print(f"buckets   {json.dumps(batches.describe())}", flush=True)
    pool, share, pool_seen = None, args.pool_share, 0.0
    if args.pool:
        args.pool.mkdir(parents=True, exist_ok=True)
        pool = Pool(args.pool, args.labels, tokenizer, cfg, args.budget, [b.tokens for b in batches.buckets],
                    args.val_fraction, pool_used)
        print(pool.refresh(args.seed + micro_done) or "pool      nothing listed yet", flush=True)
        print(f"pool      {args.pool}: {share:.0%} of forward passes once it holds enough clips; "
              f"read every {args.refresh} updates", flush=True)
    print(f"quant     {args.recipe}; teacher {args.teacher_quant}; alpha {alpha}, temperature {temp}; "
          f"decoder layers {cfg.dec_layers}", flush=True)
    if micro_done == 0:
        log.append({"step": 0, "val": validate(params)})
        print(f"val       {log[-1]['val']} before training", flush=True)
    total, seen, window, started = args.steps * args.accum, 0.0, [], time.perf_counter()
    window_seen, window_started = 0.0, started  # speed per log window: the first windows include compiling
    stream = iter(batches)
    for micro in range(micro_done, total):
        # The pool's passes are spread evenly: pass m is the pool's when floor((m + 1) * share) steps up.
        from_pool = pool is not None and pool.batches is not None and int((micro + 1) * share) > int(micro * share)
        _, arrays, seconds = next(pool.batches if from_pool else stream)
        params, opt_state, parts = train_step(params, opt_state, teacher_params, *arrays)
        window.append(parts)
        seen += seconds
        window_seen += seconds
        pool_seen += seconds if from_pool else 0.0
        if (micro + 1) % args.accum:
            continue
        step = (micro + 1) // args.accum
        if pool is not None and step % args.refresh == 0:
            message = pool.refresh(args.seed + micro + 1)
            if message:
                print(message, flush=True)
        if step % 25 == 0 or step == args.steps:
            ce, kl = (float(np.mean([float(p[i]) for p in window])) for i in (0, 1))
            now = time.perf_counter()
            print(f"step      {step}/{args.steps}  ce {ce:.4f}  kl {kl:.4f}  {now - started:7.0f} s  "
                  f"{window_seen / (now - window_started):5.0f} s of audio per s", flush=True)
            window, window_seen, window_started = [], 0.0, now
        if step % args.eval_every == 0 or step == args.steps:
            log.append({"step": step, "val": validate(params), "elapsed_s": round(time.perf_counter() - started, 1),
                        "audio_hours": round(seen / 3600, 2), **({"pool_audio_hours": round(pool_seen / 3600, 2)}
                                                                  if pool is not None else {})})
            print(f"val       {log[-1]['val']} at update {step}", flush=True)
            save_state(micro + 1)
    batches.close()
    if pool is not None and pool.batches is not None:
        pool.batches.close()

    # The track follows the weights' lineage, which the student's checkpoint records (akshar.model.track_b writes B).
    track = args.track or (student_ck.run.get("akshar") or {}).get("track", "A")
    lineage = {"phase": 2, "track": "A", "research_only": "D9"} if track == "A" else {"phase": 3, "track": "B"}
    run = {**student_ck.run, "mode": "full", "akshar": {
        **lineage, "student": args.student, "teacher": args.teacher,
        "teacher_quant": args.teacher_quant, "sources": args.sources, "labels": args.labels, "recipe": args.recipe,
        **({"validation_only": args.val_only} if args.val_only else {}), **({"weights": weights} if weights else {}),
        **({"pool": {"share": share, "folders": len(pool.used), "hours": round(sum(pool.used.values()), 1),
                     "hours_by_source": pool.by_source()}} if pool is not None else {}),
        "layers": cfg.dec_layers, "alpha": alpha, "temperature": temp, "steps": args.steps, "budget": args.budget,
        "accum": args.accum, "lr": args.lr, "seed": args.seed}}
    trained = checkpoint.Checkpoint(jax.tree.map(np.asarray, params), cfg, run, step=args.steps)
    checkpoint.save(args.out / "model.safetensors", trained)
    info = export.write(trained, args.out / f"model-{args.recipe}.cact", RECIPES[args.recipe])
    summary = {"finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), **run["akshar"],
               "data": stats, "train_clips": len(train), "val_clips": len(val), "buckets": batches.describe(),
               "log": log, "gpu_peak_bytes": (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use"),
               "export": {k: v for k, v in info.items() if k != "path"}}
    (args.out / "train.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"wrote     {info['path']} ({info['bytes'] / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
