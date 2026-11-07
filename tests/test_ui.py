import json
import random
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from akshar.eval.score import oiwer_counts, word_counts
from akshar.ui import results, server


def _sides(alignment):
    ops = alignment["ops"]
    return [r for op, r, _ in ops if op != "i"], [h for op, _, h in ops if op != "d"]


def test_text_alignment_matches_the_harness_and_keeps_every_word():
    rng = random.Random(0)
    words = "ek do teen chaar paanch laptop accha".split()
    for _ in range(200):
        ref = " ".join(rng.choices(words, k=rng.randint(0, 8)))
        hyp = " ".join(rng.choices(words, k=rng.randint(0, 8)))
        alignment = results.text_alignment(ref, hyp)
        counts = word_counts(ref, hyp)
        assert (alignment["errors"], alignment["units"]) == (counts.errors, counts.units)
        assert alignment["errors"] == sum(op != "c" for op, _, _ in alignment["ops"])
        assert _sides(alignment) == (ref.split(), hyp.split())


def test_text_alignment_normalises_both_sides():
    alignment = results.text_alignment("Ram, the KING.", "ram the king")
    assert alignment["errors"] == 0 and alignment["units"] == 3


def test_lattice_alignment_agrees_with_the_monsoon_scorer():
    clip = {"text": "इम्प्रूव होता है", "lattice": [["इम्प्रूव", "इंप्रूव", "improve"], ["होता", "hota"], ["है", "hai"]]}
    for hyp in ("improve होता है", "अब इम्प्रूव है", "", "इम्प्रूव होता है जी"):
        alignment = results.alignment("monsoon", clip, hyp)
        counts = oiwer_counts(hyp, clip["lattice"])
        assert (alignment["errors"], alignment["units"]) == (counts.errors, counts.units)
        assert alignment["errors"] == sum(op != "c" for op, _, _ in alignment["ops"])


def test_silence_counts_every_word_written():
    assert results.alignment("silence", {"text": ""}, "")["errors"] == 0
    alignment = results.alignment("silence", {"text": ""}, "धन्यवाद, जी")
    assert (alignment["errors"], alignment["units"]) == (2, 0)


def test_folder_stamps_are_dropped_from_labels():
    assert results.STAMP.sub("", "shruti_hi_20261031-192529") == "shruti"
    assert results.STAMP.sub("", "shruti_auto_first300_20261031-204141") == "shruti"
    assert results.STAMP.sub("", "akshar-v0") == "akshar-v0"


def _run(folder, language="hi", limit=None, complete=True, model="a" * 64, value=20.0):
    folder.mkdir(parents=True)
    meta = {"model": f"/m/{folder.name}.cact", "model_sha256": model, "model_bytes": 29_500_000, "language": language,
            "limit": limit, "decoding": "needle engine defaults (5 beams)", "machine": {"cpu": "test cpu"},
            "engine": {"library_sha256": "b" * 64, "build": "3.3.3"}}
    scores = {"complete": complete, "normaliser": "nfc-lower-punct2space-v1", "headline": {"value": value}}
    (folder / "run.json").write_text(json.dumps(meta))
    (folder / "scores.json").write_text(json.dumps(scores))


def test_only_complete_protocol_runs_count(tmp_path):
    _run(tmp_path / "fleurs_hi" / "akshar-v0")
    _run(tmp_path / "fleurs_hi" / "partial", limit=300, model="c" * 64)
    _run(tmp_path / "fleurs_hi" / "english-flag", language="en", model="d" * 64)
    _run(tmp_path / "fleurs_hi" / "unfinished", complete=False, model="e" * 64)
    assert [run.label for run in results.complete_runs(tmp_path, "fleurs_hi")] == ["akshar-v0"]


def test_one_row_per_model_and_engine(tmp_path):
    _run(tmp_path / "fleurs_hi" / "shruti_hi_20261031-192529", value=20.4)
    _run(tmp_path / "fleurs_hi" / "shruti_hi_20261101-080000", value=20.36)  # the same model, scored again later
    _run(tmp_path / "lahaja" / "shruti_hi_20261031-201643", value=26.31)
    table = results.summary(tmp_path)
    assert len(table["rows"]) == 1
    row = table["rows"][0]
    assert row["label"] == "shruti" and row["cells"] == {"fleurs_hi": 20.36, "lahaja": 26.31}
    assert row["mean"] is None  # four speech sets are missing


def _serve(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("TestHandler", (server.PageHandler,), {"app": app}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def page(tmp_path):
    httpd, url = _serve(server.PageApp(tmp_path / "no-models"))
    yield url
    httpd.shutdown()
    httpd.server_close()


def _get(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers




def test_the_page_and_its_errors(page):
    status, body, _ = _get(page + "/")
    assert status == 200 and b"<title>akshar</title>" in body
    status, body, _ = _get(page + "/api/models")
    assert status == 200 and json.loads(body)["models"] == []
    for path in ("/api/set?root=nowhere&set=mucs", "/api/eval-audio?set=../../etc&id=passwd", "/api/nothing"):
        status, body, _ = _get(page + path)
        assert status == 404 and "error" in json.loads(body)
    request = urllib.request.Request(page + "/api/audio", data=b"not audio", method="POST")
    try:
        urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        assert error.code == 415
    else:
        raise AssertionError("bytes that are not audio were accepted")
