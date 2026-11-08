"""Length-bucketed, streaming batches of training clips, read from FLAC on disk by background threads.

needle's trainer pads every clip to 30 s and 320 tokens and holds all audio in memory. Here clips fall into a few
length buckets; each bucket pads only to its own audio and token limits, and its batch size keeps the audio per
batch about constant. A handful of fixed shapes means a handful of compilations, then no recompiling.

A clip is (audio path, duration, token ids). Batches are numpy arrays shaped like needle's trainer expects:
wav [B, samples], lengths [B], tokens [B, max_tokens + 1].

Clips streamed in during a run (akshar.train.distill --pool) reuse the shapes of the run's fixed data: `tokens` fixes
each bucket's token length, and a clip with more tokens than its bucket holds is dropped, so nothing recompiles.
"""
from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..eval import audio

SAMPLE_RATE = 16000
BUCKETS = (6.0, 12.0, 18.0, 30.0)  # upper edges in seconds


@dataclass(frozen=True)
class Clip:
    path: Path
    seconds: float
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class Bucket:
    seconds: float  # every clip in it is at most this long
    batch: int
    tokens: int  # padded token length, including the shifted target

    @property
    def samples(self) -> int:
        return int(self.seconds * SAMPLE_RATE)


def plan(clips: list[Clip], budget_seconds: float, max_tokens: int,
         tokens: list[int] | None = None) -> tuple[list[Bucket], list[list[Clip]]]:
    """Assign clips to buckets and size each bucket's batch to about `budget_seconds` of padded audio. With `tokens`,
    each bucket's token length is given rather than fitted, and clips that do not fit it are left out."""
    groups: list[list[Clip]] = [[] for _ in BUCKETS]
    for clip in clips:
        if clip.seconds > BUCKETS[-1]:
            raise ValueError(f"{clip.path} is {clip.seconds:.1f} s, longer than the {BUCKETS[-1]:.0f} s limit")
        b = next(i for i, edge in enumerate(BUCKETS) if clip.seconds <= edge)
        if tokens is None or len(clip.tokens) <= tokens[b]:
            groups[b].append(clip)
    buckets = []
    for b, (edge, group) in enumerate(zip(BUCKETS, groups)):
        longest = max((len(c.tokens) for c in group), default=2)
        width = tokens[b] if tokens else min(max_tokens + 1, -(-longest // 16) * 16)  # a multiple of 16 shares shapes
        buckets.append(Bucket(edge, max(1, int(budget_seconds // edge)), width))
    return buckets, groups


def _assemble(bucket: Bucket, clips: list[Clip], pad: int):
    wav = np.zeros((bucket.batch, bucket.samples), np.float32)
    lengths = np.zeros(bucket.batch, np.int32)
    tokens = np.full((bucket.batch, bucket.tokens), pad, np.int32)
    for row, clip in enumerate(clips):
        samples = audio.load(clip.path)[: bucket.samples]
        wav[row, : len(samples)], lengths[row] = samples, len(samples)
        tokens[row, : len(clip.tokens)] = clip.tokens
    return wav, lengths, tokens


class Batches:
    """Endless shuffled batches over all buckets: an iterator of (bucket index, (wav, lengths, tokens), seconds).

    It is its own iterator, so every `iter()` or `next()` shares one position. Memory is bounded: at most
    `depth` batches wait to be built, `workers` are being built and `depth` wait to be taken. `close()` stops its
    threads and lets go of the batches they built. A batch whose audio cannot be read is skipped, with a message.
    """

    def __init__(self, clips: list[Clip], budget_seconds: float, max_tokens: int, pad: int, seed: int = 0,
                 workers: int = 3, depth: int = 8, tokens: list[int] | None = None):
        self.buckets, self.groups = plan(clips, budget_seconds, max_tokens, tokens)
        self.dropped = len(clips) - sum(len(g) for g in self.groups)
        if not any(len(g) >= b.batch for b, g in zip(self.buckets, self.groups)):
            raise ValueError("no bucket has enough clips for one batch")
        self.pad, self.rng = pad, np.random.default_rng(seed)
        self._jobs: queue.Queue = queue.Queue(maxsize=depth)
        self._ready: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._held: dict = {}
        self._want = 0
        threading.Thread(target=self._schedule, daemon=True).start()
        for _ in range(workers):
            threading.Thread(target=self._work, daemon=True).start()

    def _epoch(self):
        batches = []
        for b, (bucket, group) in enumerate(zip(self.buckets, self.groups)):
            order = self.rng.permutation(len(group))
            for i in range(0, len(order) - bucket.batch + 1, bucket.batch):  # drop the ragged tail, as an epoch should
                batches.append((b, [group[j] for j in order[i:i + bucket.batch]]))
        return [batches[i] for i in self.rng.permutation(len(batches))]

    def _put(self, q: queue.Queue, item) -> bool:
        while not self._stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _schedule(self):
        n = 0
        while not self._stop.is_set():
            for b, clips in self._epoch():
                if not self._put(self._jobs, (n, b, clips)):
                    return
                n += 1

    def _work(self):
        while not self._stop.is_set():
            try:
                n, b, clips = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                item = (n, b, _assemble(self.buckets[b], clips, self.pad), sum(c.seconds for c in clips))
            except Exception as error:  # an unreadable file loses its batch, not the run
                if self._stop.is_set():
                    return
                print(f"skipped   a batch: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                item = (n, b, None, 0.0)
            if not self._put(self._ready, item):
                return

    def __iter__(self):
        return self

    def __next__(self):
        # Batches come out in schedule order, so a run is reproducible whichever worker finishes first.
        while True:
            while self._want not in self._held:
                n, b, arrays, seconds = self._ready.get()
                self._held[n] = (b, arrays, seconds)
            b, arrays, seconds = self._held.pop(self._want)
            self._want += 1
            if arrays is not None:
                return b, arrays, seconds

    def close(self):
        self._stop.set()
        self._held.clear()
        for q in (self._ready, self._jobs):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def describe(self) -> list[dict]:
        return [{"max_seconds": b.seconds, "batch": b.batch, "tokens": b.tokens, "clips": len(g),
                 "hours": round(sum(c.seconds for c in g) / 3600, 2)} for b, g in zip(self.buckets, self.groups)]
