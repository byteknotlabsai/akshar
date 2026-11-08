import numpy as np
import soundfile as sf

from akshar.train.loader import Batches, Clip, plan


def _clips(tmp_path, seconds_list):
    clips = []
    for i, seconds in enumerate(seconds_list):
        path = tmp_path / f"{i}.flac"
        sf.write(path, np.full(int(seconds * 16000), 0.1 * (i + 1), np.float32), 16000, format="FLAC")
        clips.append(Clip(path, seconds, tuple(range(2, 2 + 3 * (i + 1)))))
    return clips


def test_plan_sizes_batches_by_bucket_length(tmp_path):
    clips = _clips(tmp_path, [2, 5, 7, 11, 17, 25])
    buckets, groups = plan(clips, budget_seconds=60, max_tokens=320)
    assert [b.batch for b in buckets] == [10, 5, 3, 2]
    assert [len(g) for g in groups] == [2, 2, 1, 1]
    assert all(b.tokens % 16 == 0 and b.tokens <= 321 for b in buckets)


def test_plan_rejects_clips_longer_than_the_last_bucket(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="longer than"):
        plan([Clip(tmp_path / "x.flac", 31.0, (2, 3))], budget_seconds=60, max_tokens=320)


def test_batches_have_fixed_shapes_and_a_reproducible_order(tmp_path):
    clips = _clips(tmp_path, [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])

    def first_batches(seed):
        batches = Batches(clips, budget_seconds=12, max_tokens=320, pad=0, seed=seed, workers=3)
        out = [next(batches)]
        out += [next(iter(batches)) for _ in range(5)]  # iter() shares the position: no batch is lost or awaited
        batches.close()
        return out

    a, b = first_batches(seed=1), first_batches(seed=1)
    for (ba, (wav, lengths, tokens), seconds), (bb, arrays_b, _) in zip(a, b):
        assert ba == bb == 0 and wav.shape == (2, 6 * 16000) and tokens.shape[1] % 16 == 0
        assert np.array_equal(wav, arrays_b[0]) and 0 < seconds <= 12
        assert (lengths > 0).all() and lengths.max() <= 6 * 16000


def test_fixed_token_lengths_drop_what_does_not_fit(tmp_path):
    from akshar.train.loader import Clip, plan
    clips = [Clip(tmp_path / "a.flac", 3.0, tuple(range(10))), Clip(tmp_path / "b.flac", 4.0, tuple(range(40))),
             Clip(tmp_path / "c.flac", 10.0, tuple(range(40)))]
    buckets, groups = plan(clips, 24.0, 320, tokens=[32, 64, 64, 64])
    assert [b.tokens for b in buckets] == [32, 64, 64, 64]
    assert [len(g) for g in groups] == [1, 1, 0, 0]  # b has 40 tokens, more than its bucket's 32
