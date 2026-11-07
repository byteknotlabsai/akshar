import json

import numpy as np

from akshar.data import nonspeech
from akshar.eval import audio


def _rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_generate_is_reproducible_quiet_and_never_the_test_length(tmp_path):
    nonspeech.generate(tmp_path / "a", clips=24)
    nonspeech.generate(tmp_path / "b", clips=24)
    rows = _rows(tmp_path / "a" / "manifest.jsonl")
    assert len(rows) == 24 and {r["kind"] for r in rows} == set(nonspeech.KINDS)
    assert {r["language"] for r in rows} == set(nonspeech.LANGUAGES)
    for r in rows:
        assert r["text"] == "" and r["duration"] != nonspeech.EVAL_SECONDS
        samples = audio.load(tmp_path / "a" / r["audio"])
        assert np.array_equal(samples, audio.load(tmp_path / "b" / r["audio"]))
        assert len(samples) == round(r["duration"] * audio.SAMPLE_RATE)
        if r["kind"] == "silence":
            assert r["level_db"] is None and not samples.any()
        else:
            level = 20 * np.log10(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            assert nonspeech.LEVELS_DB[0] <= r["level_db"] <= nonspeech.LEVELS_DB[1]
            assert abs(level - r["level_db"]) < 0.5
    labels = _rows(tmp_path / "a" / "labels-shruti.jsonl")
    assert [x["id"] for x in labels] == [r["id"] for r in rows]
    assert all(x["teacher"] == "" and x["by"] == "construction" for x in labels)


def test_trainer_keeps_empty_targets(tmp_path, monkeypatch):
    from akshar.train import distill

    nonspeech.generate(tmp_path / "train" / "nonspeech", clips=8)
    monkeypatch.setattr(distill, "DATA_ROOT", tmp_path)

    class Cfg:
        languages, bos, eos, max_tokens = ("en", "hi"), 1, 2, 320

        def tag(self, language):
            return 10 + self.languages.index(language)

    class Tokenizer:
        def encode(self, text):
            return [5] * len(text.split())

    train, val, stats = distill.load_clips(["nonspeech"], "shruti", Tokenizer(), Cfg(), 0.0)
    assert stats["nonspeech"]["clips"] == stats["nonspeech"]["empty_target"] == 8 and not val
    assert all(c.tokens in ((1, 10, 2), (1, 11, 2)) for c in train)
