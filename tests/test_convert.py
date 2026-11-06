import hashlib
import json

import pytest

from akshar.data import convert
from akshar.eval.normalise import normalise

VAANI = [
    "<insect_noise> इस फोटो {photo} मे बहुत सारा पानी भरा हुआ नज़र आ रहा है एक व्यक्ति का सर दिख रहा है उसके साइड "
    "{side} मे बहुत सारी नाव खड़ी हुई नज़र आ रही ",
    "<noise> इस चित्र में एक बड़े से जंगल की तस्वीर है जहां पे एक छोटा सा गुहा है सीढ़िया हैं। [breathing] </noise>",
    "<noise> दीवारों का रंग जो है वो पीला और काला है [breathing] बहुत सारे लाइटो और लैम्पो {lamp} से सजाया गया है नीचे "
    "जमीन रंग जो है काला है। </noise>",
]


def test_vaani_examples():
    assert [convert.strip_markup(text, "vaani_hi") for text in VAANI] == [
        "इस photo मे बहुत सारा पानी भरा हुआ नज़र आ रहा है एक व्यक्ति का सर दिख रहा है उसके side मे बहुत सारी नाव खड़ी "
        "हुई नज़र आ रही",
        "इस चित्र में एक बड़े से जंगल की तस्वीर है जहां पे एक छोटा सा गुहा है सीढ़िया हैं।",
        "दीवारों का रंग जो है वो पीला और काला है बहुत सारे लाइटो और lamp से सजाया गया है नीचे जमीन रंग जो है काला है।",
    ]
    # [breathing] after है is a sound, not है's English spelling: Vaani spells words in braces.
    assert convert.to_mixed(VAANI[1], "vaani_hi", {}) == normalise(
        "इस चित्र में एक बड़े से जंगल की तस्वीर है जहां पे एक छोटा सा गुहा है सीढ़िया हैं")
    assert convert.strip_markup("वहाँ [lip_rattle] गए", "vaani_hi") is None  # an unknown sound: not guessed
    assert convert.strip_markup("<noise> कुछ [unclear] </noise>", "vaani_hi") is None


@pytest.mark.parametrize("source, text, expected", [
    # IndicVoices: verbatim, English words in Devanagari; its only markup marks words nobody could make out.
    ("indicvoices_hi", "मैडम हम कल आएँगे", "मैडम हम कल आएँगे"),
    ("indicvoices_hi", "हम <unintelligible> कल आएँगे", None),
    # Rural Women: noise tags, and each English word spelled in square brackets after its Devanagari form.
    ("rural_women_hi", "<Persistent-noise-start>रूर्बन [rurban] कलस्टर [cluster] <breathing> योजना<noise>",
     "rurban cluster योजना"),
    ("rural_women_hi", "आर [R] सी [C] एच [H] आई [I] डी [D] सिक्स [Six] ज़ीरो [Zero]", "r c h i d six zero"),
    ("rural_women_hi", "बेबी [baby] को म्यूजिक [music] पसंद है", "baby को music पसंद है"),  # words, not sounds
    ("rural_women_hi", "<uh-huh> ई-रिक्सा [E-rickshaw] से <motorcycle>आई [I]", "e-rickshaw से i"),
    ("rural_women_hi", "थैंक यू [thank you] <breathing>", None),  # which words does it spell?
    ("rural_women_hi", "<sneezing> [High] स्कूल [school]", None),  # no Devanagari word before the spelling
    ("rural_women_hi", "केवाईसी [KYC} करवाना", None),
    ("rural_women_hi", "<bird_squwak] ढेर सारे खिलौने <talking>", None),
    ("rural_women_hi", "<Persistent-noise-एंड> घर", None),
    ("rural_women_hi", "<uh-huh>$ नवासी चौबीस", None),
    ("rural_women_hi", "घर तो <unintelligible> है", None),
    # Kathbath: clean Devanagari read speech.
    ("kathbath_hi", "चिकित्सकों ने जेसीओ की मौत हार्ट अटैक से बताई", "चिकित्सकों ने जेसीओ की मौत हार्ट अटैक से बताई"),
    # FLEURS: punctuation is left to the normaliser; parentheses are not.
    ("fleurs_hi_train", "सोचें। फिर, यात्रा करें!", "सोचें। फिर, यात्रा करें!"),
    ("fleurs_hi_train", "विश्व स्वास्थ्य संगठन (डब्ल्यूएचओ) ने कहा", None),
    # MUCS: narrators' scripts with digits, code in braces and spoken symbols in angle brackets.
    ("mucs_train", "इस tutorial में 2 चरण हैं", None),
    ("mucs_train", "awk में {print $5} लिखें", None),
    ("mucs_train", "<स्पेस> दबाएँ", None),
    # NPTEL: digits, symbols, slash and part-word annotations.
    ("nptel_hi", "M 1m का केंद्र", None),
    ("nptel_hi", "ω का मान", None),
    ("nptel_en", "this is the [be/between] case", None),
    ("nptel_en", "the solu [tion] is", None),
    ("nptel_en", "so we [laughter] go on", "so we go on"),
    # LibriSpeech: uppercase verbatim English without markup.
    ("librispeech_100", "I'LL SEE ABOUT IT", "I'LL SEE ABOUT IT"),
])
def test_strip_markup(source, text, expected):
    assert convert.strip_markup(text, source) == expected


def test_to_mixed():
    lexicon = {"मोबाइल": "mobile", "टाइम": "time"}
    assert convert.to_mixed("मेरा मोबाइल, टाइम पर!", "indicvoices_hi", lexicon) == "मेरा mobile time पर"
    assert convert.to_mixed("<Persistent-noise-start> स्टेट [State] बैंक [bank] में", "rural_women_hi", {}) == \
        "state bank में"
    assert convert.to_mixed("I'LL SEE ABOUT IT", "librispeech_100", {}) == "i ll see about it"  # as Shruti writes it
    assert convert.to_mixed("<noise> [breathing] </noise>", "vaani_hi", lexicon) is None  # nothing left
    assert convert.to_mixed("हम <unintelligible> गए", "indicvoices_hi", lexicon) is None


def test_align_lists_every_position():
    assert convert.align("a b c d".split(), "a x c d e".split()) == [
        ("equal", "a", "a"), ("sub", "b", "x"), ("equal", "c", "c"), ("equal", "d", "d"), ("ins", None, "e")]
    assert convert.align(["a", "b"], []) == [("del", "a", None), ("del", "b", None)]


def test_lexicon_counts_exact_matches_so_hindi_words_stay_out():
    # Shruti writes आई ("came") as "i" in 6 of its 36 places and कम ("less") as "come" in 6 of 36: neither enters.
    pairs = [("वो घर आई", "वो घर आई")] * 30 + [("वो घर आई", "वो घर i")] * 6 \
        + [("थोड़ा कम है", "थोड़ा कम है")] * 30 + [("थोड़ा कम है", "थोड़ा come है")] * 6 \
        + [("मेरा मोबाइल खो गया", "मेरा mobile खो गया")] * 5 + [("मेरा मोबाइल खो गया", "मेरा मोबाइल खो गया")]
    seen, latin = convert.count_alignments(pairs)
    assert (seen["आई"], latin["आई"]["i"], seen["कम"], latin["कम"]["come"]) == (36, 6, 36, 6)
    assert convert.learn_lexicon(pairs) == {"मोबाइल": "mobile"}  # 5 of its 6 places
    assert convert.learn_lexicon([("मेरा मोबाइल", "मेरा mobile")] * 4) == {}  # too few


def test_lexicon_skips_split_words_single_letters_and_ambiguous_words():
    pairs = [("अपना पिनकोड लिखो", "अपना pin code लिखो")] * 10 + [("एफ़ आई आर", "f i r")] * 10 \
        + [("जी सर", "जी sir")] * 10 + [("लूडो खेलो", "ludo खेलो")] * 10
    assert convert.learn_lexicon(pairs) == {"लूडो": "ludo"}


@pytest.mark.parametrize("source, rule", [
    ("indicvoices_hi", "reference"), ("kathbath_full", "reference"), ("kathbath_hi", "reference"),
    ("rural_women_full", "reference"), ("rural_women_hi", "reference"), ("vaani_hi", "reference"),
    ("fleurs_hi_train", "reference"), ("librispeech_100", "reference"),
    ("mucs_train", "shruti"), ("nptel_en", "shruti"), ("nptel_hi", "shruti"), ("shrutilipi_hi", "shruti"),
    ("a_new_source", "shruti"), ("nonspeech", "construction"),
])
def test_policy(source, rule):
    assert convert.policy(source) == rule


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _clip(source, n, text, duration=3.0):
    return {"id": f"{source}/{n}", "audio": f"clips/{n}.flac", "duration": duration, "text": text, "language": "hi",
            "source": source}


def _labels(clips):
    return [{"id": c["id"], "teacher": f"shruti {c['id']}", "language": "hi", "decode_s": 0.1} for c in clips]


@pytest.fixture
def lexicon(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "DATA_ROOT", tmp_path)
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"words": {"मोबाइल": ["mobile", 9, 10]}}, ensure_ascii=False), encoding="utf-8")
    return path


def test_convert_chooses_per_clip_by_each_rows_source(tmp_path, lexicon):
    iv = [_clip("indicvoices_hi", 0, "मेरा मोबाइल"), _clip("indicvoices_hi", 1, "हम <unintelligible> गए"),
          _clip("indicvoices_hi", 2, "बहुत लंबा", duration=31.0), _clip("indicvoices_hi", 3, "लेबल नहीं है"),
          _clip("indicvoices_hi", 4, "<unintelligible>")]
    _write(tmp_path / "train" / "iv03" / "manifest.jsonl", iv)  # a slice of IndicVoices under its own folder name
    _write(tmp_path / "train" / "iv03" / "labels-shruti.jsonl", _labels(iv[:3]))
    mucs = [_clip("mucs_train", 0, "इस tutorial में"), _clip("mucs_train", 1, "बिना label")]
    _write(tmp_path / "train" / "mucs_train" / "manifest.jsonl", mucs)
    _write(tmp_path / "train" / "mucs_train" / "labels-shruti.jsonl", _labels(mucs[:1]))
    silence = [{"id": "nonspeech/silence-0000", "audio": "clips/silence-0000.wav", "duration": 2.0, "text": "",
                "language": "en", "source": "nonspeech"}]
    _write(tmp_path / "train" / "nonspeech" / "manifest.jsonl", silence)
    _write(tmp_path / "train" / "nonspeech" / "labels-shruti.jsonl",
           [{"id": "nonspeech/silence-0000", "teacher": "", "language": "en", "by": "construction"}])

    meta = convert.convert("iv03", {"मोबाइल": "mobile"}, lexicon)
    assert _rows(tmp_path / "train" / "iv03" / "labels-mix.jsonl") == [
        {"id": "indicvoices_hi/0", "teacher": "मेरा mobile", "language": "hi", "by": "reference"},
        {"id": "indicvoices_hi/1", "teacher": "shruti indicvoices_hi/1", "language": "hi", "by": "shruti"},
        {"id": "indicvoices_hi/3", "teacher": "लेबल नहीं है", "language": "hi", "by": "reference"},  # no label needed
    ]
    assert meta["counts"] == {"over_30s": 1, "reference": 2, "reference_rejected": 2, "shruti": 1,
                              "skipped_unlabelled": 1}
    assert meta["policy"] == {"indicvoices_hi": "reference"}
    assert meta["lexicon_sha256"] == hashlib.sha256(lexicon.read_bytes()).hexdigest()
    assert json.loads((tmp_path / "train" / "iv03" / "labels-mix.json").read_text(encoding="utf-8")) == meta

    convert.convert("mucs_train", {"मोबाइल": "mobile"}, lexicon)
    assert _rows(tmp_path / "train" / "mucs_train" / "labels-mix.jsonl") == [
        {"id": "mucs_train/0", "teacher": "shruti mucs_train/0", "language": "hi", "by": "shruti"}]
    convert.convert("nonspeech", {}, lexicon)
    assert _rows(tmp_path / "train" / "nonspeech" / "labels-mix.jsonl") == [
        {"id": "nonspeech/silence-0000", "teacher": "", "language": "en", "by": "construction"}]


def test_trainer_reads_the_mix(tmp_path, lexicon, monkeypatch):
    from akshar.train import distill

    clips = [_clip("indicvoices_hi", 0, "मेरा मोबाइल"), _clip("indicvoices_hi", 1, "बहुत लंबा", duration=31.0),
             _clip("indicvoices_hi", 2, "लेबल नहीं है"), _clip("indicvoices_hi", 3, "<unintelligible>")]
    _write(tmp_path / "train" / "iv03" / "manifest.jsonl", clips)
    _write(tmp_path / "train" / "iv03" / "labels-shruti.jsonl", _labels(clips[:1]))
    convert.convert("iv03", {}, lexicon)

    class Cfg:
        languages, bos, eos, max_tokens = ("en", "hi"), 1, 2, 320

        def tag(self, language):
            return 10 + self.languages.index(language)

    class Tokenizer:
        def encode(self, text):
            return [5] * len(text.split())

    monkeypatch.setattr(distill, "DATA_ROOT", tmp_path)
    _, _, stats = distill.load_clips(["iv03"], "mix", Tokenizer(), Cfg(), 0.0)
    assert stats["iv03"] == {"clips": 2, "empty_target": 0, "unlabelled": 1, "over_30s": 1, "other_language": 0}


def test_convert_without_shruti_labels(tmp_path, lexicon):
    _write(tmp_path / "train" / "vaani_hi" / "manifest.jsonl",
           [_clip("vaani_hi", 0, VAANI[0]), _clip("vaani_hi", 1, "<noise> कुछ [unclear] </noise>")])
    meta = convert.convert("vaani_hi", {}, lexicon)
    rows = _rows(tmp_path / "train" / "vaani_hi" / "labels-mix.jsonl")
    assert [(r["id"], r["by"], r["language"]) for r in rows] == [("vaani_hi/0", "reference", "hi")]
    assert meta["counts"] == {"reference": 1, "reference_rejected": 1, "skipped_unlabelled": 1}
    assert meta["shruti_labels"] is None and "no labels-shruti.jsonl" in meta["note"]

    _write(tmp_path / "train" / "shrutilipi_hi" / "manifest.jsonl", [_clip("shrutilipi_hi", 0, "कुछ भी")])
    meta = convert.convert("shrutilipi_hi", {}, lexicon)
    assert (tmp_path / "train" / "shrutilipi_hi" / "labels-mix.jsonl").read_text(encoding="utf-8") == ""
    assert meta["rows"] == 0 and meta["counts"] == {"skipped_unlabelled": 1}
    assert "Shruti-only clips get none" in meta["note"]


def test_cli_learns_writes_and_reloads_the_lexicon(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "DATA_ROOT", tmp_path)
    clips = [_clip("indicvoices_hi", n, "मेरा मोबाइल खो गया" if n < 6 else "वो घर आई") for n in range(12)]
    labels = [{"id": c["id"], "teacher": "मेरा mobile खो गया" if n < 6 else "वो घर आई", "language": "hi"}
              for n, c in enumerate(clips)]
    _write(tmp_path / "train" / "iv01" / "manifest.jsonl", clips + [_clip("indicvoices_hi", 12, "मोबाइल")])
    _write(tmp_path / "train" / "iv01" / "labels-shruti.jsonl", labels)  # the last clip has no label: not learned from
    path = tmp_path / "lexicon.json"
    convert.main(["iv01", "--lexicon", str(path), "--learn"])
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["sources"] == ["iv01"] and body["words"] == {"मोबाइल": ["mobile", 6, 6]}
    assert convert.load_lexicon(path) == {"मोबाइल": "mobile"}
    rows = _rows(tmp_path / "train" / "iv01" / "labels-mix.jsonl")
    assert [r["teacher"] for r in rows[::6]] == ["मेरा mobile खो गया", "वो घर आई", "mobile"]
    assert {r["by"] for r in rows} == {"reference"}
