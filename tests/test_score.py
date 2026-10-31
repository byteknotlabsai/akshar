import random

import jiwer
import pytest

from akshar.eval.score import char_counts, score_clips, word_counts


def test_word_counts():
    counts = word_counts("a b c d", "a x c d e")
    assert (counts.sub, counts.dele, counts.ins, counts.units) == (1, 0, 1, 4)


def test_corpus_rate_is_total_errors_over_total_reference_words():
    clips = [{"text": "a b", "hyp": "a"}, {"text": "a b c d e f g h", "hyp": "a b c d e f g h"}]
    scores = score_clips(clips, "wer")
    assert (scores.wer.errors, scores.wer.units) == (1, 10)
    assert scores.wer.rate == pytest.approx(10.0)  # not 25.0, the mean of the per-clip rates


def test_normaliser_is_applied_to_both_sides():
    scores = score_clips([{"text": "Ram, the KING.", "hyp": "ram the king"}], "wer")
    assert scores.wer.errors == 0


def test_empty_reference_counts_insertions():
    scores = score_clips([{"text": "", "hyp": "hello there"}, {"text": "a", "hyp": "a"}], "wer")
    assert (scores.wer.ins, scores.wer.units) == (2, 1)


def test_char_counts_include_spaces():
    counts = char_counts("ab cd", "abcd")
    assert (counts.dele, counts.units) == (1, 5)


def test_total_errors_match_jiwer():
    rng = random.Random(7)
    words = ["है", "हैं", "में", "मैं", "laptop", "की", "के", "order"]
    refs = [" ".join(rng.choices(words, k=rng.randint(1, 12))) for _ in range(300)]
    hyps = [" ".join(rng.choices(words, k=rng.randint(0, 12))) for _ in range(300)]
    ours = score_clips([{"text": r, "hyp": h} for r, h in zip(refs, hyps)], "wer").wer
    theirs = jiwer.process_words(refs, hyps)
    assert ours.errors == theirs.substitutions + theirs.deletions + theirs.insertions
    assert ours.units == theirs.hits + theirs.substitutions + theirs.deletions


def test_silence_counts_words():
    clips = [{"text": "", "hyp": ""}, {"text": "", "hyp": "धन्यवाद"}, {"text": "", "hyp": "thank you so much"}]
    scores = score_clips(clips, "silence")
    assert (scores.silence_words, scores.silence_clips_with_words) == (5, 2)


def test_oiwer_accepts_any_listed_spelling():
    lattice = [["इम्प्रूव", "इंप्रूव", "improve"], ["होता", "hota"], ["है", "hai"]]
    clip = {"text": "इम्प्रूव होता है", "lattice": lattice}
    assert score_clips([{**clip, "hyp": "improve होता है"}], "oiwer").oiwer.errors == 0
    assert score_clips([{**clip, "hyp": "improve होती है"}], "oiwer").oiwer.errors == 1
