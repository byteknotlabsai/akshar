"""The gated sets' preparers, run on tiny parquet files with the column layout their dataset cards document."""
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from akshar.eval import prepare as prep


def _wav(seconds=0.5):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(16000 * seconds), np.float32), 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(prep, "EVAL_ROOT", tmp_path / "eval")
    return tmp_path


def _prepare(name, roots, monkeypatch, columns):
    raw = roots / "raw" / name / "data" / "test-00000.parquet"
    raw.parent.mkdir(parents=True)
    pq.write_table(pa.table(columns), raw)
    monkeypatch.setattr(prep, "download", lambda ts: [raw])
    out = prep.prepare(name)
    with open(out / "manifest.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_lahaja_clips_of_one_recording_get_their_own_ids(roots, monkeypatch):
    rows = _prepare("lahaja", roots, monkeypatch, {
        "audio_filepath": [{"bytes": _wav(), "path": "534_f960_chunk_0_subchunk_1.wav"},
                           {"bytes": _wav(), "path": "534_f960_chunk_0_subchunk_2.wav"}],
        "text": ["नमस्ते दुनिया", "दो किताबें"], "lang": ["hi", "hi"],
        "verbatim": ["नमस्ते दुनिया", "दो किताबें"], "normalized": ["नमस्ते दुनिया", "2 किताबें"],
        "duration": [0.5, 0.5], "fname": ["534", "534"],
    })
    assert [r["id"] for r in rows] == ["534_f960_chunk_0_subchunk_1", "534_f960_chunk_0_subchunk_2"]
    assert rows[1]["text"] == "दो किताबें" and rows[1]["lahaja_normalized"] == "2 किताबें"
    assert rows[1]["lahaja_recording"] == "534"
    assert rows[0]["audio"] == "audio/534_f960_chunk_0_subchunk_1.wav" and rows[0]["duration"] == 0.5


def test_svarah_ids_come_from_the_audio_file_name(roots, monkeypatch):
    rows = _prepare("svarah", roots, monkeypatch, {
        "audio_filepath": [{"bytes": _wav(), "path": "spk1/utt_0001.wav"}, {"bytes": _wav(), "path": None}],
        "duration": [0.5, 0.5], "text": ["hello there", "order my groceries"],
    })
    assert [r["id"] for r in rows] == ["utt_0001", "svarah-00001"]
    assert rows[1]["text"] == "order my groceries"
