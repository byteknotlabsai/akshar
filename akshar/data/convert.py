"""Turn datasets' own transcripts into `mixed`-style targets, and choose per clip between them and Shruti's labels.

    python -m akshar.data.convert indicvoices_hi rural_women_full --lexicon ~/akshar-data/lexicon.json --learn
    python -m akshar.data.convert vaani_hi mucs_train --lexicon ~/akshar-data/lexicon.json

A reference becomes a target in three steps. strip_markup removes noise tags and, where a dataset spells an English
word out in brackets after its Devanagari form (Rural Women's रूर्बन [rurban], Vaani's फोटो {photo}), keeps only the
spelling; it rejects a reference with unknown words, digits, code or broken markup. The normaliser then removes
punctuation. Last, a lexicon turns English words written in Devanagari into Latin, as Shruti writes them. It is
learned where Shruti's labels align with the references: a word enters only if Shruti writes one Latin word for most
of its aligned occurrences, exact matches included, so a Hindi word that Shruti only sometimes hears as English
stays out (आई, "came", is sometimes written "i").

Each source has a policy. Reference first: the converted reference, which needs no Shruti label, or else Shruti's
label. Shruti only: NPTEL's references are not verbatim, MUCS's are narrators' scripts with digits, and unknown
sources are not trusted. Clips over 30 s are skipped, as the trainer drops them. Writes, next to the manifest:
labels-mix.jsonl (id, teacher, language, by: reference, shruti or construction), read by the trainer with
`--labels mix`, and labels-mix.json (counts, policy, lexicon hash). `--labels-name fixed` writes labels-fixed.*
instead, so that a re-conversion can sit beside the labels a run is reading.

D29 (the label audit of 2026-11-08): a lexicon word is also found under a folded spelling (no nukta, short for long
i, e for ai, o for au, anusvara for a half nasal, no reph), English number words convert with any lexicon, Hindi
number slips (नो for नौ, तेतीस) are repaired, English sources fold accents, and NPTEL drops speaker labels, rejects
slashes, slide-time tags, the motto and numbers split across segments, reads Indian grouping (1,05,000) and years.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from ..eval.normalise import VERSION, normalise
from ..eval.paths import DATA_ROOT

MAX_SECONDS = 30.0
REFERENCE_FIRST = {"indicvoices_hi", "kathbath_full", "kathbath_hi", "rural_women_full", "rural_women_hi", "vaani_hi",
                   "fleurs_hi_train", "librispeech_100", "librispeech_360", "nptel_en", "nptel_en_b",
                   "cv_hi", "cv_en_sa"}  # Common Voice: read sentences, without digits
# English references whose numbers are written in digits: they are spelled out as words first (D28).
SPELL_NUMBERS = {"nptel_en", "nptel_en_b"}
# English sources, whose targets need no Hindi repairs.
ENGLISH = {"nptel_en", "nptel_en_b", "cv_en_sa", "librispeech_100", "librispeech_360"}
# Read English whose accented Latin letters (Castellón) are folded to plain ones instead of rejecting the reference.
# Not NPTEL: nearly every accented letter in its subtitles is mojibake (PrimÕs, becomeÉ), which stays rejected.
FOLD_ACCENTS = {"cv_en_sa", "librispeech_100", "librispeech_360"}
# Marks for words the annotator could not make out: the reference has a hole, so it is never a target.
# (Vaani also writes [intelligible] and [intangible] for [unintelligible].)
UNKNOWN = {"unintelligible", "unk", "unknown", "inaudible", "unclear", "indistinct", "incomprehensible", "foreign",
           "intelligible", "intangible"}
# Sounds marked in square brackets (Vaani's [breathing], Rural Women's [horn], NPTEL's [laughter]), matched with
# their words joined by _. <Tags> go whatever they name. The longer list is what Vaani and Rural Women write (D29).
EVENTS = set("""breath breathing inhaling exhaling sigh yawn yawning cough coughing sneeze sneezing sniff sniffing
    throat_clearing lip_smack lipsmack smack click tsk swallowing laugh laughing laughter cry crying noise
    background_noise insect_noise static hum music singing horn bell beep door dishes animal barking talking
    background_talking whispering child baby
    inhale exhale static_noise people_talking bird_sound bird_squawk bird_squawking bird_noise lip_smacks
    lip_smacking lips_smack lips_smacking lips_smash lip_lick lipsticks child_talking child_yelling child_crying
    child_laughing children_talking children_yelling baby_talking baby_crying phone_ringing ringing yelling
    nose_blowing animal_sound motorcycle vehicle_sound vehicle_noise vehicle_horn music_noise door_closing clinking
    clink thumping uhh ugh hmm stammers stammering person_stammering snorting buzz tapping faltering swallowing
    tongue_clicks throat_cleaning scratching siren vocalized_noise whaling pause""".split())
# Annotators' misspelt English glosses, mapped to the spelling the same word has elsewhere (label audit, D29).
GLOSS_TYPOS = {"rop": "rope", "meckup": "makeup", "charted": "chartered", "accountance": "accountants",
               "icecream": "ice cream", "colddrink": "cold drink", "docter": "doctor", "receipe": "recipe",
               "cack": "cake", "moter": "motor", "protin": "protein", "tripal": "triple"}
# Hindi words that annotators gloss in Latin as if they were English (लाख [Lakh]): they stay in Devanagari.
GLOSS_KEEP = {unicodedata.normalize("NFC", w) for w in ("लाख", "करोड़", "करोड़", "हज़ार", "हजार", "सौ", "पनीर", "पिस्ता")}
# English number words in their usual Devanagari spellings, converted with any lexicon so that one number is never
# split across scripts (एटीन thousand). वन, सेवन, टू, फोर and एट are left out: they are also Hindi or other English words.
NUMBER_WORDS = {"थ्री": "three", "फाइव": "five", "सिक्स": "six", "नाइन": "nine", "टेन": "ten", "इलेवन": "eleven",
                "ट्वेल्व": "twelve", "थर्टीन": "thirteen", "फोर्टीन": "fourteen", "फिफ्टीन": "fifteen",
                "सिक्सटीन": "sixteen", "सेवनटीन": "seventeen", "एटीन": "eighteen", "नाइनटीन": "nineteen",
                "ट्वेंटी": "twenty", "ट्वेन्टी": "twenty", "थर्टी": "thirty", "फोर्टी": "forty", "फिफ्टी": "fifty",
                "सिक्सटी": "sixty", "सेवंटी": "seventy", "सेवेंटी": "seventy", "एटी": "eighty", "नाइंटी": "ninety",
                "हंड्रेड": "hundred", "थाउज़ेंड": "thousand", "थाउजेंड": "thousand", "मिलियन": "million",
                "सेकंड": "second", "सेकेंड": "second", "थर्ड": "third", "फोर्थ": "fourth", "फिफ्थ": "fifth",
                "परसेंट": "percent", "पर्सेंट": "percent", "पर्सेन्ट": "percent"}
# Hindi number words: नो beside one of them is नौ (nine), not English "no" (D29).
HINDI_NUMBERS = {unicodedata.normalize("NFC", w) for w in """शून्य एक दो तीन चार पाँच पांच छह छः सात आठ नौ दस ग्यारह
    बारह तेरह चौदह पंद्रह पन्द्रह सोलह सत्रह अठारह उन्नीस बीस तीस चालीस पचास साठ सत्तर अस्सी नब्बे सौ हज़ार हजार लाख करोड़
    ज़ीरो जीरो डबल zero double""".split()}
# Words after which NPTEL's 1100-1999 is a year (in 1995: nineteen ninety five), not a count.
YEAR_WORDS = {"in", "since", "year", "from", "till", "until", "by", "during", "of", "before", "after", "around"}
INDIAN_GROUPED = re.compile(r"\d{1,2}(?:,\d{2})+,\d{3}")  # 1,05,000: lakh and crore grouping
LAST_WORD = re.compile(r"([A-Za-z]+)\W*$")
ENDS_IN_DIGIT = re.compile(r"\d[\s.,;:]*$")
LATIN_ACCENTED = re.compile(r"[\u00c0-\u024f]")
# Spelling differences a lexicon lookup tolerates (D29): a nukta, long/short i, ai/e, au/o (बौक्स/बॉक्स), a half
# nasal for an anusvara (पर्सेन्ट/परसेंट) and the reph.
NUKTA = "\u093c"
FOLD_SIGNS = str.maketrans({"ई": "इ", "ी": "ि", "ऐ": "ए", "ै": "े", "औ": "ऑ", "ौ": "ॉ"})
HALF_NASAL = re.compile(r"[नमणञङ]\u094d(?=[\u0915-\u0939])")
REPH = "र\u094d"
# The bracket in which a dataset spells the Devanagari word before it in English; its other brackets are markup.
GLOSS = {"rural_women": "[]", "vaani": "{}"}
# Rural Women's gloss bracket also holds a few sounds. Only these are sounds there: [baby], [music], [dishes] and
# [motorcycle] are glosses of the word before them (बेबी [baby]), so EVENTS cannot decide (label audit, D29).
RURAL_SOUNDS = {"clink", "clinking", "uhh", "ugh", "hmm", "horn", "thumping", "stammers", "stammering"}
# Hindi words that are also English words (head or sir, just or bus): a speaker may mean either, so they stay.
AMBIGUOUS = {"सर", "बस"}
DEVANAGARI_WORD = re.compile(r"[\u0900-\u097f\u200c\u200d]+(?:-[\u0900-\u097f\u200c\u200d]+)*")
LATIN_WORD = re.compile(r"[a-z]+")
TAG = re.compile(r"<\s*/?[A-Za-z][A-Za-z_\-]*\s*>")
TAG_SPACED = re.compile(r"<\s*/?[A-Za-z][A-Za-z_\- ]*\s*>")  # Vaani's and Rural Women's <static noise>; NPTEL's <g n u n> is maths
MARKED = re.compile(r"[<\[{]\s*/?([^<>\[\]{}]*?)\s*[>\]}]")
SQUARE = re.compile(r"\[\s*([A-Za-z_]+)\s*\]")
SQUARE_ANY = re.compile(r"\[[^\[\]]*\]")
GLOSS_GROUP = {"{}": re.compile(r"(\{[^{}]*\})"), "[]": re.compile(r"(\[[^\[\]]*\])")}
DEVANAGARI_ANY = re.compile(r"[\u0900-\u097f]")
# A spoken token a gloss may replace: Devanagari, with the _ of ए_सी {A_C}, the dots of ए.टी.एम {ATM}, a hyphen or
# an apostrophe (डॉन'ट {don't}) inside.
DEVANAGARI_TOKEN = re.compile(r"[\u0900-\u097f\u200c\u200d_\-.'’]+")
VOWELS = re.compile(r"[aeiouy]")
# Hindi function words: a gloss of several words never spans one, so और मोरक्कोनऑइल [moroccon oil] replaces one token
# (D29). A one-word gloss may still spell one (रूम की {key}, ओ [O] के [K]).
STOPWORDS = {unicodedata.normalize("NFC", w) for w in """और से का की के में पर है हैं हूँ हूं था थी थे को ने एक ये यह वो
    वह तो भी ही जो कि कुछ बहुत सब अपना अपने अपनी मेरा मेरी मेरे हमारा हमारी हमारे उसका उसकी उसके इसका इसकी इसके इस उस या
    फिर अब तब यहाँ वहाँ यहां वहां ओर हे हें""".split()}
# Sounds that Vaani's annotators sometimes write in the gloss braces (है {breathing}): never a word's spelling.
BRACE_SOUNDS = {"inhale", "inhaling", "exhale", "exhaling", "breath", "breathing", "noise", "static", "static_noise",
                "background_noise", "pause", "cough", "coughing", "sneeze", "sneezing", "sniff", "sniffing", "laugh",
                "laughing", "laughter", "hmm", "uhh", "umm", "ugh", "lip_smack", "lipsmack", "lip_smacking",
                "throat_clearing", "swallowing", "tsk"}
GLOSS_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
# NPTEL segments with no single spoken form: slide-time tags, the sung motto, and maths slashes (1/r is "by" or "over").
NPTEL_REJECT = re.compile(r"refer\s+(?:slide\s+)?time|paramam|knowledge is supreme", re.IGNORECASE)
SLASH = re.compile(r"(?:\b\w{1,3}|\d)\s*/\s*(?:\w{1,3}\b|\d)")
# A transcript's leading speaker label ("Jayanta Chatterjee:", "Student:"), which nobody says.
SPEAKER = re.compile(r"^\s*(?:(?:Prof\.?|Professor|Dr\.?|Students?|Participant|Moderator|Interviewer|Host)"
                     r"(?:\s+[A-Z][A-Za-z.]*){0,3}|[A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*){1,3})\s*:\s*")


def _gloss_pattern(brackets: str) -> re.Pattern:
    open_, close = map(re.escape, brackets)
    return re.compile(f"({DEVANAGARI_WORD.pattern})\\s*{open_}\\s*([A-Za-z]+(?:['’-][A-Za-z]+)*)\\s*{close}")


GLOSS_PATTERNS = {brackets: _gloss_pattern(brackets) for brackets in GLOSS.values()}


def _trusted(ch: str) -> bool:
    if ch in "<>[]{}()" or ch.isdigit():
        return False
    return ch.isspace() or "\u0900" <= ch <= "\u097f" or ch in "\u200c\u200d" or ch.isascii() and ch.isalpha() \
        or unicodedata.category(ch).startswith("P")


def _is_event(inner: str) -> bool:
    return "_".join(inner.lower().split()) in EVENTS


def _latin_gloss(gloss: str) -> list[str] | None:
    """A gloss's English words, lowercased: letters joined by _ spell one word (A_T_M is atm, ice_cream ice cream);
    known misspellings are corrected. None if it is not plain English."""
    words = []
    for token in gloss.split():
        parts = [part for part in token.split("_") if part]
        if not parts or not all(GLOSS_WORD.fullmatch(part) for part in parts):
            return None
        joined = "".join(parts) if all(len(part) == 1 for part in parts) else " ".join(parts)
        for word in joined.lower().split():
            words.extend(GLOSS_TYPOS.get(word, word).split())
    return words or None


def _same_word(a: str, b: str) -> bool:
    """Whether two Latin spellings are of one word: equal without vowels and doubled letters (color and colour,
    resturant and restaurant, said and side), unlike cap and cab, or lights and light."""
    def skeleton(word: str) -> str:
        return re.sub(r"(.)\1+", r"\1", VOWELS.sub("", word))
    return skeleton(a) == skeleton(b)


def _span(out: list[str], n: int) -> int:
    """How many of the last n tokens a gloss of n words replaces: the run of Devanagari tokens before the gloss,
    which beyond its first token holds no function word (और मोरक्कोनऑइल [moroccon oil] replaces one token). 0 if
    the token before the gloss is not Devanagari."""
    k = 0
    for token in reversed(out[-n:]):
        word = token.strip(".,!?।\"'")
        if not word or not DEVANAGARI_TOKEN.fullmatch(word) or k and normalise(word).strip() in STOPWORDS:
            break
        k += 1
    return k


def _apply_glosses(text: str, brackets: str, lexicon: dict[str, str] | Lexicon | None) -> str | None:
    """Put each gloss in place of the Devanagari words it spells. A gloss of n words replaces up to n words before it
    (टी शर्ट {T shirt} is t shirt; see _span); Devanagari in the gloss bracket is a corrected spelling (शारी {सारी} is
    सारी).
    A one-word English gloss of a word the lexicon knows takes the lexicon's spelling when the two are spellings of
    one word (कलर {color} is colour, साइड {said} is side; कैप {cap} stays cap beside the lexicon's cab), and a
    Hindi word glossed as if English keeps its Devanagari (लाख [Lakh]). A sound in the gloss bracket (Rural Women's
    [horn], RURAL_SOUNDS; Vaani's {breathing}, BRACE_SOUNDS) is dropped, keeping the word before it. None when a gloss
    has no Devanagari word to replace."""
    out: list[str] = []
    for part in GLOSS_GROUP[brackets].split(text):
        if not (len(part) >= 2 and part[0] == brackets[0] and part[-1] == brackets[1]):
            out.extend(part.split())
            continue
        inner = part[1:-1].strip()
        if not inner or (brackets == "[]" and inner.lower() in RURAL_SOUNDS) \
                or (brackets == "{}" and "_".join(inner.lower().split()) in BRACE_SOUNDS):
            continue
        correction = bool(DEVANAGARI_ANY.search(inner))
        new = inner.split() if correction else _latin_gloss(inner)
        if new is None:
            return None
        n = _span(out, len(inner.split()))
        if n == 0:
            return None
        spoken = [token.strip(".,!?।\"'") for token in out[-n:]]
        del out[-n:]
        if n == 1 and not correction:
            key = normalise(spoken[0]).strip()
            if key in GLOSS_KEEP:
                new = spoken
            elif lexicon and key in lexicon and _same_word(" ".join(new), lexicon[key]):
                new = lexicon[key].split()
        out.extend(new)
    return " ".join(out)


def strip_markup(text: str, source: str, lexicon: dict[str, str] | Lexicon | None = None) -> str | None:
    """A reference's words without markup, or None if it cannot be trusted as a target.

    Removes <tags> and </tags> (with spaces inside too for Vaani and Rural Women), and sound events in square brackets. In Vaani every
    square bracket is a sound and the curly bracket holds glosses; in Rural Women the square bracket holds glosses and
    sounds both (see _apply_glosses). None for an unknown-word mark, digits, anything left in brackets (MUCS's {code},
    NPTEL's [be/between], a gloss with no Devanagari word before it) and symbols other than punctuation."""
    if any(m.group(1).lower() in UNKNOWN for m in MARKED.finditer(text)):
        return None
    family = next((name for name in GLOSS if source.startswith(name)), None)
    text = (TAG_SPACED if family else TAG).sub(" ", text)
    if family == "vaani":
        text = _apply_glosses(SQUARE_ANY.sub(" ", text), "{}", lexicon)
    elif family == "rural_women":
        text = _apply_glosses(text, "[]", lexicon)
    else:
        text = SQUARE.sub(lambda m: " " if _is_event(m.group(1)) else m.group(0), text)
    if text is None or not all(_trusted(ch) for ch in text):
        return None
    return " ".join(text.split())


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Levenshtein alignment of two word lists: (op, reference word, hypothesis word) per position, op one of equal,
    sub, del (a reference word the hypothesis lacks; hypothesis word None) and ins (an extra hypothesis word)."""
    ops = []
    for block in Levenshtein.opcodes(ref, hyp):
        for r, h in zip_longest(ref[block.src_start:block.src_end], hyp[block.dest_start:block.dest_end]):
            op = "equal" if block.tag == "equal" else "sub" if r is not None and h is not None else \
                "del" if r is not None else "ins"
            ops.append((op, r, h))
    return ops


def count_alignments(pairs) -> tuple[Counter, dict[str, Counter]]:
    """How often each Devanagari reference word is aligned to a word of the label, and to which Latin words.

    A substitution beside an inserted Latin word is not counted as a spelling: Shruti split the word (पिनकोड, "pin
    code"), and only half of it is aligned."""
    seen, latin = Counter(), defaultdict(Counter)
    for reference, label in pairs:
        ops = align(normalise(reference).split(), normalise(label).split())
        inserted = [op == "ins" and bool(LATIN_WORD.fullmatch(h)) for op, _, h in ops] + [False]
        for k, (op, r, h) in enumerate(ops):
            if op not in ("equal", "sub") or not DEVANAGARI_WORD.fullmatch(r):
                continue
            seen[r] += 1
            if LATIN_WORD.fullmatch(h) and not inserted[k - 1] and not inserted[k + 1]:
                latin[r][h] += 1
    return seen, latin


def choose(seen: Counter, latin: dict[str, Counter], min_count: int = 5, min_share: float = 0.8) -> dict:
    """Devanagari word -> (Latin word, times Shruti wrote it, times the word was aligned at all).

    Single letters stay out: a letter's name is often a Hindi word too (पी, ए, ओ, सी), which only the words around
    it can tell apart, and converting some letters of an acronym but not others is worse than converting none."""
    chosen = {}
    for word, counts in latin.items():
        spelling, count = counts.most_common(1)[0]
        if count >= min_count and count / seen[word] >= min_share and len(spelling) > 1 and word not in AMBIGUOUS:
            chosen[word] = (spelling, count, seen[word])
    return chosen


def learn_lexicon(pairs, min_count: int = 5, min_share: float = 0.8) -> dict[str, str]:
    """Devanagari word -> the Latin word Shruti writes for it, from (reference, Shruti's label) pairs.

    References must have been through strip_markup. Both sides are normalised and aligned. A word is kept when one
    Latin spelling accounts for at least min_count of its aligned occurrences and at least min_share of all of them,
    exact matches included."""
    return {word: entry[0] for word, entry in choose(*count_alignments(pairs), min_count, min_share).items()}


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
            "twelve": "twelfth"}
# A number: Indian grouping, Western grouping or plain digits, with a decimal part and a % or ordinal suffix. Digits
# joined to a letter, a dot or a stray comma (x2, 1,0,5000) are not matched, so strip_markup rejects the line.
NUMBER = re.compile(r"(?<![\w.,])(\d{1,2}(?:,\d{2})+,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%|st|nd|rd|th)?(?!\w|,\d)")


def english_cardinal(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + english_cardinal(n % 100))
    for value, word in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= value:
            head, rest = divmod(n, value)
            return english_cardinal(head) + " " + word + ("" if rest == 0 else " " + english_cardinal(rest))
    raise ValueError(n)


def english_ordinal(n: int) -> str:
    *head, last = english_cardinal(n).split()
    last = ORDINALS.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join([*head, last])


def indian_cardinal(n: int) -> str:
    """A number written with Indian grouping, in lakh and crore: 1,05,000 -> one lakh five thousand."""
    parts = []
    for value, word in ((10 ** 7, "crore"), (10 ** 5, "lakh")):
        if n >= value:
            head, n = divmod(n, value)
            parts.append(english_cardinal(head) + " " + word)
    if n or not parts:
        parts.append(english_cardinal(n))
    return " ".join(parts)


def english_year(n: int) -> str:
    """A year from 1100 to 1999 as it is said: nineteen ninety five, nineteen hundred, nineteen oh five."""
    head, rest = divmod(n, 100)
    if rest == 0:
        return f"{ONES[head]} hundred"
    return f"{ONES[head]} {'oh ' + ONES[rest] if rest < 10 else english_cardinal(rest)}"


def spell_english_numbers(text: str) -> str:
    """Numbers in digits as English words: 3 -> three, 1,500 -> one thousand five hundred, 1,05,000 -> one lakh five
    thousand, 2.5 -> two point five, 10% -> ten percent, 2nd -> second. 1100-1999 is a year after a word such as in,
    since or from (in 1995 -> in nineteen ninety five) and is left in digits elsewhere, as is 2010-2099 after such a
    word (twenty fifteen or two thousand fifteen?); digits attached to letters (x2, 3d) and malformed groupings
    (1,0,5000) are left too, so strip_markup rejects those lines."""
    def say(m: re.Match) -> str:
        digits = m.group(1)
        whole = int(digits.replace(",", ""))
        if whole >= 10 ** 12:
            return m.group(0)
        if m.group(3) in ("st", "nd", "rd", "th") and not m.group(2):
            return f" {english_ordinal(whole)} "
        if INDIAN_GROUPED.fullmatch(digits):
            words = indian_cardinal(whole)
        elif 1100 <= whole <= 2099 and "," not in digits and not m.group(2) and not m.group(3):
            before = LAST_WORD.search(text, 0, m.start())
            after_year_word = bool(before) and before.group(1).lower() in YEAR_WORDS
            if whole < 2000 and not after_year_word or whole >= 2010 and after_year_word:
                return m.group(0)  # a count that may be a year, or a year said two ways: not guessed
            words = english_year(whole) if whole < 2000 else english_cardinal(whole)
        else:
            words = english_cardinal(whole)
        if m.group(2):
            words += " point " + " ".join(ONES[int(d)] for d in m.group(2))
        return f" {words}{' percent' if m.group(3) == '%' else ''} "
    return NUMBER.sub(say, text)


def fold_accents(text: str) -> str:
    """Accented Latin letters as plain ones (Castellón is castellon); a letter with no plain form (ø, ß) stays."""
    def plain(m: re.Match) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFKD", m.group(0)) if not unicodedata.combining(ch))
    return LATIN_ACCENTED.sub(plain, text)


def fold_devanagari(word: str) -> str:
    """A spelling-tolerant key for a Devanagari word (see FOLD_SIGNS)."""
    word = unicodedata.normalize("NFC", unicodedata.normalize("NFD", word).replace(NUKTA, ""))
    return HALF_NASAL.sub("\u0902", word).replace(REPH, "र").translate(FOLD_SIGNS)


class Lexicon:
    """A lexicon (Devanagari word -> Latin spelling) with its lookup tables, built once per conversion. A non-empty
    lexicon also converts the English number words; the folded table holds every entry under its folded key, minus
    keys that fold together with different spellings."""

    def __init__(self, words: dict[str, str]):
        self.words = words
        self.exact = {**NUMBER_WORDS, **words} if words else {}
        self.folded: dict[str, str] = {}
        clashes = set()
        for word, spelling in self.exact.items():
            key = fold_devanagari(word)
            if self.folded.setdefault(key, spelling) != spelling:
                clashes.add(key)
        for key in clashes:
            del self.folded[key]

    def __bool__(self) -> bool:
        return bool(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def __getitem__(self, word: str) -> str:
        return self.words[word]

    def convert(self, word: str) -> str:
        """The Latin spelling for a normalised word, found exactly or under its folded spelling, else the word."""
        if word in self.exact:
            return self.exact[word]
        if DEVANAGARI_ANY.search(word):
            return self.folded.get(fold_devanagari(word), word)
        return word


def convert_word(word: str, lexicon: dict[str, str] | Lexicon) -> str:
    return (lexicon if isinstance(lexicon, Lexicon) else Lexicon(lexicon)).convert(word)


def fix_hindi_numbers(words: list[str]) -> list[str]:
    """Transcribers' slips in Hindi number words: नो beside a number word is नौ (nine), not English "no", and तेतीस
    is तैंतीस (D29)."""
    out = ["तैंतीस" if word == "तेतीस" else word for word in words]
    changed = True
    while changed:  # a repaired नौ makes its neighbour a number word too (चार नो नो)
        changed = False
        for k, word in enumerate(out):
            if word == "नो" and any(0 <= j < len(out) and out[j] in HINDI_NUMBERS for j in (k - 1, k + 1)):
                out[k], changed = "नौ", True
    return out


def to_mixed(text: str, source: str, lexicon: dict[str, str] | Lexicon) -> str | None:
    """A reference as a `mixed`-style target, or None if it cannot be one. A conversion of many rows passes a Lexicon,
    built once; a plain dict is wrapped per call.

    NPTEL's English (SPELL_NUMBERS) first loses its speaker label and spells its numbers out; a segment with a slash
    between short symbols (1/r), a slide-time tag or the sung motto, or one that starts with a digit (the tail of a
    number split at its decimal point), is not a target. Read English (FOLD_ACCENTS) folds accents. After strip_markup
    and the normaliser, a lexicon turns English words written in Devanagari into Latin, and Hindi number slips are
    repaired."""
    if source in SPELL_NUMBERS:
        if NPTEL_REJECT.search(text) or SLASH.search(text) or text.lstrip()[:1].isdigit():
            return None
        text = spell_english_numbers(SPEAKER.sub("", text, count=1))
    if source in FOLD_ACCENTS:
        text = fold_accents(text)
    if not isinstance(lexicon, Lexicon):
        lexicon = Lexicon(lexicon)
    stripped = strip_markup(text, source, lexicon)
    if stripped is None:
        return None
    words = [lexicon.convert(word) for word in normalise(stripped).split()]
    if source not in ENGLISH:
        words = fix_hindi_numbers(words)
    return " ".join(words).lower() or None


def policy(source: str) -> str:
    """Where a source's targets come from: reference (falling back to Shruti's label), shruti, or construction."""
    if source == "nonspeech":
        return "construction"
    return "reference" if source in REFERENCE_FIRST else "shruti"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _shruti_labels(folder: Path) -> dict[str, dict] | None:
    path = folder / "labels-shruti.jsonl"
    return {row["id"]: row for row in _read_jsonl(path)} if path.exists() else None


def _pairs(names: list[str]) -> list[tuple[str, str]]:
    """(reference after strip_markup, Shruti's label) for every Hindi clip of these sources that has both."""
    pairs = []
    for name in names:
        folder = DATA_ROOT / "train" / name
        labels = _shruti_labels(folder) or {}
        for row in _read_jsonl(folder / "manifest.jsonl"):
            label = labels.get(row["id"])
            reference = strip_markup(row.get("text") or "", row.get("source") or name)
            if label is not None and reference and row["language"] == "hi":
                pairs.append((reference, label["teacher"]))
    return pairs


def write_lexicon(path: Path, names: list[str], min_count: int = 5, min_share: float = 0.8) -> dict[str, str]:
    """Learn a lexicon from these sources and write it as JSON: word -> [Latin word, count, aligned occurrences]."""
    chosen = choose(*count_alignments(_pairs(names)), min_count, min_share)
    body = {"sources": names, "min_count": min_count, "min_share": min_share,
            "words": {word: list(chosen[word]) for word in sorted(chosen)}}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {word: entry[0] for word, entry in chosen.items()}


def load_lexicon(path: Path) -> dict[str, str]:
    return {word: entry[0] for word, entry in json.loads(path.read_text(encoding="utf-8"))["words"].items()}


def split_numbers(rows: list[dict]) -> set[str]:
    """Ids of NPTEL segments that end in a digit while the next segment of the same video starts with one: the
    subtitles were cut at full stops, decimal points included, so the number straddles the two and its spoken
    "point" is in neither. Ids are <source>/<video>_<start seconds>."""
    by_video: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for row in rows:
        if (row.get("source") or "") in SPELL_NUMBERS:
            video, _, start = row["id"].rpartition("_")
            try:
                by_video[video].append((float(start), row))
            except ValueError:
                continue
    split = set()
    for segments in by_video.values():
        segments.sort(key=lambda s: s[0])
        for (_, head), (_, tail) in zip(segments, segments[1:]):
            if ENDS_IN_DIGIT.search(head.get("text") or "") and (tail.get("text") or "").lstrip()[:1].isdigit():
                split.add(head["id"])
    return split


def convert(name: str, lexicon: dict[str, str], lexicon_path: Path, labels_name: str = "mix") -> dict:
    """Write labels-<labels_name>.jsonl and .json for one source folder. The policy follows each manifest row's
    `source`, so a slice such as iv02 (rows from indicvoices_hi) gets its dataset's policy."""
    folder = DATA_ROOT / "train" / name
    labels = _shruti_labels(folder)
    counts, policies, rows = Counter(), {}, []
    table = Lexicon(lexicon)
    manifest = _read_jsonl(folder / "manifest.jsonl")
    split = split_numbers(manifest)
    for row in manifest:
        if row["duration"] > MAX_SECONDS:
            counts["over_30s"] += 1
            continue
        source = row.get("source") or name
        rule = policies.setdefault(source, policy(source))
        label = (labels or {}).get(row["id"])
        target = None
        if rule == "reference" and row["id"] in split:
            counts["split_number"] += 1
        elif rule == "reference":
            target = to_mixed(row.get("text") or "", source, table)
        if rule == "reference" and target is None:
            counts["reference_rejected"] += 1
        if rule == "construction":
            out = {"teacher": "", "language": label["language"] if label else row["language"], "by": "construction"}
        elif target is not None:
            out = {"teacher": target, "language": label["language"] if label and label["language"] else row["language"],
                   "by": "reference"}
        elif label is not None:
            teacher = label["teacher"]
            if label["language"] == "hi":
                teacher = " ".join(fix_hindi_numbers(teacher.split()))
            out = {"teacher": teacher, "language": label["language"], "by": "shruti"}
        else:
            counts["skipped_unlabelled"] += 1
            continue
        counts[out["by"]] += 1
        rows.append({"id": row["id"], **out})
    (folder / f"labels-{labels_name}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {"source": name, "rows": len(rows), "counts": dict(sorted(counts.items())), "policy": policies,
            "shruti_labels": "labels-shruti.jsonl" if labels is not None else None, "normaliser": VERSION,
            "lexicon": str(lexicon_path), "lexicon_sha256": hashlib.sha256(lexicon_path.read_bytes()).hexdigest()}
    if labels is None:
        meta["note"] = ("no labels-shruti.jsonl, so every clip is unlabelled: only converted references and "
                        "constructed targets get rows, and Shruti-only clips get none")
    (folder / f"labels-{labels_name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                                                        encoding="utf-8")
    return meta


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.convert", description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="+", metavar="source", help="folders under $AKSHAR_DATA/train")
    parser.add_argument("--lexicon", type=Path, required=True, help="lexicon JSON to load, or to write with --learn")
    parser.add_argument("--learn", action="store_true", help="learn the lexicon from these sources first")
    parser.add_argument("--labels-name", default="mix", metavar="NAME",
                        help="write labels-NAME.jsonl and labels-NAME.json (default mix)")
    args = parser.parse_args(argv)
    lexicon = write_lexicon(args.lexicon, args.sources) if args.learn else load_lexicon(args.lexicon)
    for name in args.sources:
        print(json.dumps(convert(name, lexicon, args.lexicon, args.labels_name), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
