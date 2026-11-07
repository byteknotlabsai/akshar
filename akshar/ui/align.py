"""Word errors of one transcript against what was said, for display: the harness's normaliser and edit distance
(akshar.eval.score), with the word-by-word alignment behind them."""
from ..eval.normalise import normalise
from ..eval.score import word_counts, word_ops


def text_alignment(reference: str, hyp: str) -> dict:
    """{"metric", "errors", "units", "ops"}: ops are [op, reference word, transcript word] with op c (correct),
    s (substituted), d (a reference word the transcript lacks) or i (an extra word in the transcript)."""
    ref, out = normalise(reference), normalise(hyp)
    counts = word_counts(ref, out)
    return {"metric": "WER", "errors": counts.errors, "units": counts.units, "ops": word_ops(ref, out)}
