import unicodedata

from akshar.eval.normalise import normalise


def test_nfc_makes_nukta_forms_equal():
    # क़ as one code point (U+0958) and as क + nukta (U+0915 U+093C) must score as the same word.
    assert normalise("क़लम") == normalise("क़लम")


def test_lowercase_and_whitespace():
    assert normalise("  Hello   WORLD \n") == "hello world"


def test_punctuation_becomes_space_on_both_scripts():
    assert normalise("सोचें। फिर (लंबी) यात्रा, ठीक?") == "सोचें फिर लंबी यात्रा ठीक"
    assert normalise("gnu/linux") == "gnu linux"
    assert normalise("don't — “quoted”") == "don t quoted"
    assert normalise("॥ श्लोक ॥") == "श्लोक"


def test_devanagari_signs_survive():
    # Matras, virama, anusvara, chandrabindu and nukta are marks (M*), not punctuation.
    text = "हिंदी स्कूल माँ ज़रूर क्या"
    assert normalise(text) == unicodedata.normalize("NFC", text)


def test_digits_and_symbols_are_kept():
    # % is punctuation (Po); ₹ and + are symbols (Sc, Sm). Digits are never turned into words.
    assert normalise("₹334 + 5%") == "₹334 + 5"


def test_romanise_keeps_latin_words_and_lowercases_itrans():
    from akshar.eval.normalise import romanise

    assert romanise("तुम्हारा Laptop अच्छा है।") == "tumhara laptop achcha hai"
    # Lowercasing merges ITRANS's long and short vowels, so these two spellings meet.
    assert romanise("दिन") == romanise("दीन") == "dina"
