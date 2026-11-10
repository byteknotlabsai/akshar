import json

import pytest

from akshar.data import agreement


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _clip(folder, shard, n, text, duration=3600.0):
    return {"id": f"shrutilipi_hi/{n}", "audio": f"clips/{shard}/{n}.flac", "duration": duration, "text": text,
            "language": "hi", "source": "shrutilipi_hi"}


def _label(n, teacher):
    return {"id": f"shrutilipi_hi/{n}", "teacher": teacher, "language": "hi", "decode_s": 0.1}


@pytest.mark.parametrize("wer, name", [(0, "0"), (4.9, "<5"), (5, "5-15"), (14.9, "5-15"), (15, "15-30"),
                                       (30, "30-60"), (60, ">60"), (250, ">60")])
def test_bands(wer, name):
    assert agreement.band(wer) == name


def test_disagreement_compares_in_mixed_style_with_a_lexicon():
    lexicon = agreement.Lexicon({"बैंक": "bank"})
    assert agreement.disagreement("वो बैंक गए।", "वो bank गए", "shrutilipi_hi", lexicon) == 0.0
    assert agreement.disagreement("वो बैंक गए।", "वो bank गए", "shrutilipi_hi", None) == pytest.approx(100 / 3)
    assert agreement.disagreement("", "कुछ", "shrutilipi_hi", None) == float("inf")


def test_filter_keeps_agreeing_clips_from_several_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(agreement, "DATA_ROOT", tmp_path)
    a = [_clip("a", "s1", 0, "उन्होंने कहा कि अफवाहें फैलाना मीडिया का काम नहीं है"),  # 0%
         _clip("a", "s1", 1, "आज पुणे के पास एक कार्यक्रम में मंत्री ने कहा"),  # 1 of 9 words: 11%
         _clip("a", "s1", 2, "संक्षिप्त समाचार अब आप सुन रहे हैं"),  # the script differs wholly
         _clip("a", "s1", 3, "बिना लेबल")]
    _write(tmp_path / "train" / "a" / "manifest.jsonl", a)
    _write(tmp_path / "train" / "a" / "labels-akshar-v2.jsonl", [
        _label(0, "उन्होंने कहा कि अफवाहें फैलाना मीडिया का काम नहीं है"),
        _label(1, "आज पुणे के पास एक कार्यक्रम में मंत्री ने बोला"),
        _label(2, "कृपया कुमार कृपया कुमार कृपया कुमार")])
    b = [_clip("b", "s2", 0, "उन्होंने कहा कि अफवाहें फैलाना मीडिया का काम नहीं है"),  # the same id again: skipped
         _clip("b", "s2", 4, "मौसम का हाल", duration=1800.0)]
    _write(tmp_path / "train" / "b" / "manifest.jsonl", b)
    _write(tmp_path / "train" / "b" / "labels-akshar-v2.jsonl", [_label(0, "x"), _label(4, "मौसम का हाल")])

    report = agreement.filter_agreeing(["a", "b"], "akshar-v2", 5.0, "agree5")
    kept = _rows(tmp_path / "train" / "agree5" / "manifest.jsonl")
    assert [r["id"] for r in kept] == ["shrutilipi_hi/0", "shrutilipi_hi/4"]
    assert kept[1]["audio"] == "clips/s2/4.flac"  # audio paths are left as the source folders wrote them
    labels = _rows(tmp_path / "train" / "agree5" / "labels-mix.jsonl")
    assert labels[0] == {"id": "shrutilipi_hi/0", "teacher": a[0]["text"], "language": "hi", "by": "akshar-v2"}
    assert report["kept"] == 2 and report["kept_hours"] == 1.5
    assert report["bands"]["0"] == {"clips": 2, "hours": 1.5}
    assert report["bands"]["5-15"] == {"clips": 1, "hours": 1.0}
    assert report["bands"][">60"] == {"clips": 1, "hours": 1.0}
    assert report["skipped"] == {"duplicate": 1, "unlabelled": 1}
    assert json.loads((tmp_path / "train" / "agree5" / "agreement.json").read_text(encoding="utf-8")) == report

    loose = agreement.filter_agreeing(["a", "b"], "akshar-v2", 15.0, None)  # report only: nothing written
    assert loose["kept"] == 3 and not (tmp_path / "train" / "None").exists()


def test_cli_report_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agreement, "DATA_ROOT", tmp_path)
    _write(tmp_path / "train" / "a" / "manifest.jsonl", [_clip("a", "s1", 0, "मौसम का हाल")])
    _write(tmp_path / "train" / "a" / "labels-akshar-v2.jsonl", [_label(0, "मौसम का हाल")])
    agreement.main(["a", "--labels", "akshar-v2", "--report-only"])
    assert json.loads(capsys.readouterr().out)["kept"] == 1
    with pytest.raises(SystemExit):
        agreement.main(["a", "--labels", "akshar-v2"])  # --out missing
