"""The text normaliser used for scoring. One function, applied identically to references and hypotheses.

Shruti's harness describes its normaliser as "Unicode NFC, lowercase, punctuation removed" and publishes no code.
"Punctuation" here is every character whose Unicode general category starts with P (Pc, Pd, Pe, Pf, Pi, Po, Ps).
That covers ASCII punctuation, curly quotes and the Devanagari danda (।, ॥). Symbols (S*: + = < > $ ₹ ° and so on),
digits and Devanagari signs (matras, virama, nukta, anusvara) are kept.

Each punctuation character becomes a space, so "gnu/linux" scores as two words and "don't" as "don t", on both sides.
docs/eval.md records how this choice was checked against Shruti's published numbers.

Change this function only with a new decision in docs/decisions.md: it moves every number we report.
"""
import unicodedata

VERSION = "nfc-lower-punct2space-v1"


def is_punctuation(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if is_punctuation(ch) else ch for ch in text)
    return " ".join(text.split())


def romanise(text: str) -> str:
    """Script-agnostic form for Hinglish scoring: normalise, Devanagari to ITRANS, then lowercase.

    Latin-script words pass through unchanged. Lowercasing merges ITRANS's long and short vowels (A/a) and retroflex
    and dental consonants (T/t), so the comparison tolerates spelling. This reproduces Udgam Labs' "Romanised: ITRANS"
    MUCS score for Shruti (34.79; docs/eval.md). It is a neutral converter, not akshar's `roman` output style.
    """
    from indic_transliteration import sanscript

    return " ".join(sanscript.transliterate(normalise(text), sanscript.DEVANAGARI, sanscript.ITRANS).lower().split())
