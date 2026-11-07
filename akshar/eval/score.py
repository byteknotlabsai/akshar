"""Scorers: word and character edit distance, the Monsoon lattice scorer (OIWER), and invented words on silence.

Corpus-level rates only: all errors divided by all reference units, never an average of per-clip rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest

from rapidfuzz.distance import Levenshtein

from .normalise import normalise, romanise


@dataclass
class Counts:
    """Edit counts summed over clips. `units` is the number of reference words (or characters)."""

    sub: int = 0
    dele: int = 0
    ins: int = 0
    units: int = 0
    clips: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        """Error rate in percent. With no reference units, any insertion makes it infinite."""
        if self.units == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return 100.0 * self.errors / self.units

    def __iadd__(self, other: Counts) -> Counts:
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins
        self.units += other.units
        self.clips += other.clips
        return self

    def as_dict(self) -> dict:
        return {"rate": round(self.rate, 4), "errors": self.errors, "units": self.units, "sub": self.sub,
                "del": self.dele, "ins": self.ins, "clips": self.clips}


def _edit_counts(ref, hyp) -> Counts:
    ops = Levenshtein.editops(ref, hyp)
    counts = Counts(units=len(ref), clips=1)
    for op in ops:
        if op.tag == "replace":
            counts.sub += 1
        elif op.tag == "delete":
            counts.dele += 1
        else:
            counts.ins += 1
    return counts


def word_counts(ref: str, hyp: str) -> Counts:
    """Word-level edit distance between two already-normalised strings."""
    return _edit_counts(ref.split(), hyp.split())


def word_ops(ref: str, hyp: str) -> list[list[str]]:
    """The word alignment behind word_counts, for display: [op, reference word, hypothesis word] per position, op one of
    c (correct), s (substituted), d (a reference word the hypothesis lacks) and i (an extra hypothesis word)."""
    ref_words, hyp_words = ref.split(), hyp.split()
    ops = []
    for block in Levenshtein.opcodes(ref_words, hyp_words):
        pairs = zip_longest(ref_words[block.src_start:block.src_end], hyp_words[block.dest_start:block.dest_end],
                            fillvalue="")
        ops.extend(["c" if block.tag == "equal" else "s" if r and h else "d" if r else "i", r, h] for r, h in pairs)
    return ops


def char_counts(ref: str, hyp: str) -> Counts:
    """Character-level edit distance between two already-normalised strings.

    Characters are Unicode code points after NFC, and the single spaces between words count as characters.
    """
    return _edit_counts(ref, hyp)


def oiwer_counts(hyp: str, lattice: list[list[str]]) -> Counts:
    """Monsoon's official lattice scorer, called exactly as the dataset card shows: raw hypothesis, raw lattice."""
    from voi_oiwer import oiwer

    result = oiwer(hyp, lattice, "hindi")
    ins, dele, sub = result[4]
    return Counts(sub=sub, dele=dele, ins=ins, units=result[5], clips=1)


@dataclass
class Scores:
    wer: Counts = field(default_factory=Counts)
    cer: Counts = field(default_factory=Counts)
    oiwer: Counts = field(default_factory=Counts)
    roman: Counts = field(default_factory=Counts)
    silence_words: int = 0
    silence_clips_with_words: int = 0


def score_clips(clips: list[dict], scorer: str, romanised: bool = False) -> Scores:
    """Score a list of {"text", "hyp", optional "lattice"} dicts. scorer is "wer", "oiwer" or "silence".

    romanised also scores both sides after `romanise` (script-agnostic WER, for Hinglish sets)."""
    scores = Scores()
    for clip in clips:
        hyp = normalise(clip["hyp"])
        if scorer == "silence":
            words = len(hyp.split())
            scores.silence_words += words
            scores.silence_clips_with_words += int(words > 0)
            continue
        ref = normalise(clip["text"])
        scores.wer += word_counts(ref, hyp)
        scores.cer += char_counts(ref, hyp)
        if scorer == "oiwer":
            scores.oiwer += oiwer_counts(clip["hyp"], clip["lattice"])
        if romanised:
            scores.roman += word_counts(romanise(clip["text"]), romanise(clip["hyp"]))
    return scores
