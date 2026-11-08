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
    assert convert.strip_markup("वहाँ [lip_rattle] गए", "vaani_hi") == "वहाँ गए"  # every Vaani [bracket] is a sound
    assert convert.strip_markup("<noise> कुछ [unclear] </noise>", "vaani_hi") is None
    assert convert.strip_markup("कुछ [intelligible] यहाँ", "vaani_hi") is None  # Vaani's spelling of unintelligible


# The label audit of 2026-11-08 (D29): 9 of 10 Vaani rejections were good transcripts lost to markup parsing.
@pytest.mark.parametrize("text, expected", [
    ("<bird squawking> यह एक गोडाउन {godown} है </bird squawking>", "यह एक godown है"),  # a tag with a space
    ("<noise> ये देखो [lip smacking] कई शारी {सारी} दुकाने </noise>", "ये देखो कई सारी दुकाने"),  # a correction
    ("एक ने लाल टी शर्ट {T shirt} पहन रखी है", "एक ने लाल t shirt पहन रखी है"),  # a two-word gloss
    ("टी_वी {T_V} और एटीएम {A_T_M} है", "tv और atm है"),  # letters joined by _ spell one word, as the token before
    ("एस.बी.आई {SBI} बैंक {bank} का ए.टी.एम {ATM} है", "sbi bank का atm है"),  # an abbreviation written with dots
    ("डॉन'ट {don't} ओवरलोड {overload}", "don't overload"),
    ("साइड {said} में कैप {cap} और लाइटें {lights} रेस्टोरेंट {resturant}", "side में cap और lights restaurant"),
    ("उसपे ब्लू कलर {blue colour} का है", "उसपे blue colour का है"),
    ("रूम {room} की {key} और गूगल पे {google pay} से", "room key और google pay से"),  # glosses of function-like words
    ("बच्चे {inhale} और बर्तन है {breathing} यहाँ", "बच्चे और बर्तन है यहाँ"),  # sounds written in the gloss braces
    ("बैटल {battle} रोप {rop} का मेकअप {meckup}", "battle rope का makeup"),  # annotators' misspellings
    ("कलर {color} और कलर {colour}", "colour और colour"),  # the lexicon's spelling wins over the gloss
    ("चर्च [church] गए", "चर्च गए"),  # square brackets are sounds in Vaani, so the gloss is dropped
    ("{godown} है", None),  # no Devanagari word before the gloss
    ("बहुत {very long gloss} है", "very long gloss है"),  # a longer gloss spells the one token before it (ओलाप्लैक्स)
])
def test_vaani_markup_after_the_audit(text, expected):
    lexicon = {"कलर": "colour", "साइड": "side", "कैप": "cab", "लाइट": "light", "रेस्टोरेंट": "restaurant"}
    assert convert.strip_markup(text, "vaani_hi", lexicon) == expected


@pytest.mark.parametrize("text, expected", [
    ("सिलाई के लिए [horn] कपड़े चॉक [chalk] से", "सिलाई के लिए कपड़े chalk से"),  # a sound keeps the word before it
    ("बेबी [baby] को म्यूजिक [music] पसंद है", "baby को music पसंद है"),  # glosses, although EVENTS has both words
    ("डॉक्टर [Docter] से रेसिपी [receipe] केक [cack]", "doctor से recipe cake"),  # misspelt glosses
    ("पाँच चार लाख [Lakh] रुपये पनीर [paneer] पिस्ता [Pistaa]", "पाँच चार लाख रुपये पनीर पिस्ता"),  # Hindi words
    ("पैन [PAN] कार्ड [card <inhaling> नंबर [number]", None),  # an open bracket: still not guessed
    ("और मोरक्कोनऑइल [moroccon oil] से", "और moroccon oil से"),  # a two-word gloss of one compound token
    ("फोन [Phone] पे [Pay] या गूगल [google] पे [pay] से", "phone pay या google pay से"),
    ("विशेष रूप से ओलाप्लैक्स [ola plex] का", "विशेष रूप से ola plex का"),
    ("हमारे ब्यूटी पार्लर [ beauty parlour] में", "हमारे beauty parlour में"),
    ("ठीक है <Persi stent-noise-end>", "ठीक है"),  # a broken tag with a space inside
])
def test_rural_women_glosses_after_the_audit(text, expected):
    assert convert.strip_markup(text, "rural_women_hi") == expected


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
    ("rural_women_hi", "थैंक यू [thank you] <breathing>", "thank you"),  # a two-word gloss spells the two words before
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
    ("fleurs_hi_train", "reference"), ("librispeech_100", "reference"), ("librispeech_360", "reference"),
    ("nptel_en", "reference"),  # D28: its English references, numbers spelled out
    ("mucs_train", "shruti"), ("nptel_hi", "shruti"), ("shrutilipi_hi", "shruti"),
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


def test_lexicon_lookup_tolerates_spelling_variants():
    lexicon = {"कन्फर्म": "confirm", "ड्राइवर": "driver", "परसेंट": "percent", "बॉक्स": "box", "हेलमेट": "helmet"}
    assert convert.to_mixed("कन्फ़र्म ड्राईवर पर्सेन्ट बौक्स हैलमेट", "indicvoices_hi", lexicon) == \
        "confirm driver percent box helmet"
    assert convert.fold_devanagari("पर्सेन्ट") == convert.fold_devanagari("परसेंट")
    assert convert.fold_devanagari("कन्फ़र्म") == convert.fold_devanagari("कन्फर्म")
    # Two entries that fold together with different spellings are not guessed; the exact ones still work.
    clash = {"ड्राइवर": "driver", "ड्राईवर": "drive"}
    assert convert.to_mixed("ड्राइवर ड्राईवर ड्रा़इवर", "indicvoices_hi", clash) == "driver drive ड्रा़इवर"
    assert convert.to_mixed("कन्फ़र्म", "indicvoices_hi", {}) == "कन्फ़र्म"


def test_english_number_words_convert_with_any_lexicon():
    # One number must not come out in two scripts (एटीन thousand): the number words join every non-empty lexicon.
    assert convert.to_mixed("ये ट्वेन्टी थाउज़ेंड के बैग टेन पर्सेंट एटीन थाउज़ेंड सेकेंड थर्ड", "indicvoices_hi",
                            {"बैग": "bag"}) == "ये twenty thousand के bag ten percent eighteen thousand second third"
    assert convert.to_mixed("एटीन थाउज़ेंड", "rural_women_hi", {}) == normalise("एटीन थाउज़ेंड")  # empty: converts nothing


def test_hindi_number_slips_are_repaired():
    assert convert.to_mixed("चार ज़ीरो [zero]नो ज़ीरो [zero] पाँच पाँच नो सात", "rural_women_hi", {}) == \
        "चार zero नौ zero पाँच पाँच नौ सात"
    assert convert.to_mixed("यू नो मैं गया", "indicvoices_hi", {}) == "यू नो मैं गया"  # English "no" stays
    assert convert.to_mixed("पता तेतीस नगरानगंज", "indicvoices_hi", {}) == "पता तैंतीस नगरानगंज"
    assert convert.fix_hindi_numbers("नो नो दो".split()) == ["नौ", "नौ", "दो"]


def test_read_english_sources_fold_accents():
    assert convert.to_mixed("Mádl and Jenő in Castellón", "cv_en_sa", {}) == "madl and jeno in castellon"
    assert convert.to_mixed("Søren", "cv_en_sa", {}) is None  # no plain letter for ø: not guessed
    assert convert.to_mixed("Castellón", "indicvoices_hi", {}) is None  # only read English folds
    # NPTEL's accented letters are mojibake for quotes and dots (PrimÕs, becomeÉ), so they stay rejected.
    assert convert.to_mixed("PrimÕs algorithm", "nptel_en_b", {}) is None
    assert convert.to_mixed("the Schrödinger equation", "nptel_en_b", {}) is None


def test_nptel_rejects_what_has_no_single_spoken_form():
    assert convert.to_mixed("Jayanta Chatterjee:Just to recap the 3 components", "nptel_en_b", {}) == \
        "just to recap the three components"
    assert convert.to_mixed("Student: so the answer", "nptel_en", {}) == "so the answer"
    assert convert.to_mixed("Professor Shashi Shekhar Mishra: yes", "nptel_en", {}) == "yes"
    assert convert.to_mixed("dipole induced dipole 1/r power 6", "nptel_en_b", {}) is None  # by or over?
    assert convert.to_mixed("so dy/dx and V/I here", "nptel_en_b", {}) is None
    assert convert.to_mixed("the input/output pair", "nptel_en_b", {}) == "the input output pair"
    assert convert.to_mixed("(Refer Slide Time: 18:16) so we see", "nptel_en_b", {}) is None
    assert convert.to_mixed("Refer Time 18:16", "nptel_en_b", {}) is None
    assert convert.to_mixed("Lethman Paramam Dheyam Knowledge is supreme", "nptel_en_b", {}) is None
    assert convert.to_mixed("the last term is going to be <g n u n>", "nptel_en_b", {}) is None  # maths, not a tag
    assert convert.to_mixed("so we [laughter] go on <noise>", "nptel_en_b", {}) == "so we go on"
    assert convert.to_mixed("445 inch", "nptel_en_b", {}) is None  # the tail of a number split at its decimal point
    assert convert.to_mixed("5 ohm and if we assume", "nptel_en_b", {}) is None


def test_nptel_indian_grouping_and_years():
    assert convert.spell_english_numbers("1,05,000 and 2,50,000 and 30,77,000 and 1,00,00,000").split() == \
        "one lakh five thousand and two lakh fifty thousand and thirty lakh seventy seven thousand and one crore".split()
    assert convert.spell_english_numbers("250,000 or 10,000").split() == \
        "two hundred fifty thousand or ten thousand".split()
    assert convert.to_mixed("now 1,0,5000 rupees", "nptel_en_b", {}) is None  # a malformed grouping: not guessed
    assert convert.spell_english_numbers("1, 2, 3").split() == "one , two , three".split()
    assert convert.spell_english_numbers("in 1995 since 1900 by 1905 of 1750").split() == \
        "in nineteen ninety five since nineteen hundred by nineteen oh five of seventeen fifty".split()
    assert convert.spell_english_numbers("in 2005") == "in  two thousand five "
    assert convert.to_mixed("about 1500 rupees", "nptel_en_b", {}) is None  # fifteen hundred or one thousand ...?
    assert convert.to_mixed("in 2015 we", "nptel_en_b", {}) is None  # twenty fifteen or two thousand fifteen?
    assert convert.to_mixed("about 2050 units and 1,995 items", "nptel_en_b", {}) == \
        "about two thousand fifty units and one thousand nine hundred ninety five items"
    assert convert.to_mixed("the 1995th", "nptel_en_b", {}) == \
        "the one thousand nine hundred ninety fifth"


def test_convert_rejects_numbers_split_across_nptel_segments(tmp_path, lexicon):
    rows = [{"id": f"nptel_en_b/vid_x_{start}", "audio": f"clips/s/vid_x_{start}.flac", "duration": 3.0,
             "text": text, "language": "en", "source": "nptel_en_b"} for start, text in
            [("10.0", "the reactance will be 42"), ("12.5", "5 ohm and so on"), ("20.0", "W equal to 3"),
             ("30.0", "fine, 4 volts")]]
    _write(tmp_path / "train" / "nptel_en_b" / "manifest.jsonl", rows[::-1])  # order on disk does not matter
    meta = convert.convert("nptel_en_b", {}, lexicon, labels_name="fixed")
    assert [r["teacher"] for r in _rows(tmp_path / "train" / "nptel_en_b" / "labels-fixed.jsonl")] == \
        ["fine four volts", "w equal to three"]  # 42 | 5 ohm is 42.5; 3 before "fine" is whole
    assert meta["counts"] == {"reference": 2, "reference_rejected": 2, "skipped_unlabelled": 2, "split_number": 1}
    assert not (tmp_path / "train" / "nptel_en_b" / "labels-mix.jsonl").exists()
    assert json.loads((tmp_path / "train" / "nptel_en_b" / "labels-fixed.json").read_text(encoding="utf-8")) == meta


def test_shruti_fallback_labels_get_the_hindi_number_fix(tmp_path, lexicon):
    clips = [_clip("mucs_train", 0, "scripted")]
    _write(tmp_path / "train" / "mucs_train" / "manifest.jsonl", clips)
    _write(tmp_path / "train" / "mucs_train" / "labels-shruti.jsonl",
           [{"id": "mucs_train/0", "teacher": "चार नो सात यू नो", "language": "hi"}])
    convert.convert("mucs_train", {}, lexicon)
    assert _rows(tmp_path / "train" / "mucs_train" / "labels-mix.jsonl")[0]["teacher"] == "चार नौ सात यू नो"


def test_nptel_english_numbers_become_words():
    assert convert.to_mixed("So, power supplied by the current v 3 into 2 right", "nptel_en", {}) == \
        "so power supplied by the current v three into two right"
    assert convert.spell_english_numbers("1,500 or 2.5, 10% of the 2nd and the 21st").split() == \
        "one thousand five hundred or two point five , ten percent of the second and the twenty first".split()
    assert convert.english_cardinal(999_999) == "nine hundred ninety nine thousand nine hundred ninety nine"
    assert convert.to_mixed("x2 plus 3d", "nptel_en", {}) is None  # digits on letters are not guessed
    assert convert.to_mixed("in 1990 there were 12", "mucs_train", {}) is None  # only NPTEL's English is spelled out
