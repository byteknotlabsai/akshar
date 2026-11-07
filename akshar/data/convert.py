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
`--labels mix`, and labels-mix.json (counts, policy, lexicon hash).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from ..eval.normalise import VERSION, normalise
from ..eval.paths import DATA_ROOT

MAX_SECONDS = 30.0
REFERENCE_FIRST = {"indicvoices_hi", "kathbath_full", "kathbath_hi", "rural_women_full", "rural_women_hi", "vaani_hi",
                   "fleurs_hi_train", "librispeech_100", "librispeech_360", "nptel_en"}
# English references whose numbers are written in digits: they are spelled out as words first (D28).
SPELL_NUMBERS = {"nptel_en"}
# Marks for words the annotator could not make out: the reference has a hole, so it is never a target.
UNKNOWN = {"unintelligible", "unk", "unknown", "inaudible", "unclear", "indistinct", "incomprehensible", "foreign"}
# Sounds marked in square brackets (Vaani's [breathing], NPTEL's [laughter]). <Tags> go whatever they name.
EVENTS = set("""breath breathing inhaling exhaling sigh yawn yawning cough coughing sneeze sneezing sniff sniffing
    throat_clearing lip_smack lipsmack smack click tsk swallowing laugh laughing laughter cry crying noise
    background_noise insect_noise static hum music singing horn bell beep door dishes animal barking talking
    background_talking whispering child baby""".split())
# The bracket in which a dataset spells the Devanagari word before it in English; its other brackets are markup.
GLOSS = {"rural_women": "[]", "vaani": "{}"}
# Hindi words that are also English words (head or sir, just or bus): a speaker may mean either, so they stay.
AMBIGUOUS = {"सर", "बस"}
DEVANAGARI_WORD = re.compile(r"[\u0900-\u097f\u200c\u200d]+(?:-[\u0900-\u097f\u200c\u200d]+)*")
LATIN_WORD = re.compile(r"[a-z]+")
TAG = re.compile(r"<\s*/?[A-Za-z][A-Za-z_\-]*\s*>")
MARKED = re.compile(r"[<\[{]\s*/?([^<>\[\]{}]*?)\s*[>\]}]")
SQUARE = re.compile(r"\[\s*([A-Za-z_]+)\s*\]")


def _gloss_pattern(brackets: str) -> re.Pattern:
    open_, close = map(re.escape, brackets)
    return re.compile(f"({DEVANAGARI_WORD.pattern})\\s*{open_}\\s*([A-Za-z]+(?:['’-][A-Za-z]+)*)\\s*{close}")


GLOSS_PATTERNS = {brackets: _gloss_pattern(brackets) for brackets in GLOSS.values()}


def _trusted(ch: str) -> bool:
    if ch in "<>[]{}()" or ch.isdigit():
        return False
    return ch.isspace() or "\u0900" <= ch <= "\u097f" or ch in "\u200c\u200d" or ch.isascii() and ch.isalpha() \
        or unicodedata.category(ch).startswith("P")


def strip_markup(text: str, source: str) -> str | None:
    """A reference's words without markup, or None if it cannot be trusted as a target.

    Removes <tags> and </tags>, and sound events in square brackets unless the source spells words in them. Replaces
    a Devanagari word followed by its English spelling in the source's gloss bracket with that spelling, lowercased.
    None for an unknown-word mark, digits, anything left in brackets (MUCS's {code}, NPTEL's [be/between], a
    multi-word or misplaced gloss) and symbols other than punctuation."""
    if any(m.group(1).lower() in UNKNOWN for m in MARKED.finditer(text)):
        return None
    brackets = next((b for name, b in GLOSS.items() if source.startswith(name)), "")
    if brackets:
        text = GLOSS_PATTERNS[brackets].sub(lambda m: f" {m.group(2).lower()} ", text)
    text = TAG.sub(" ", text)
    if brackets != "[]":
        text = SQUARE.sub(lambda m: " " if m.group(1).lower() in EVENTS else m.group(0), text)
    if not all(_trusted(ch) for ch in text):
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
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%|st|nd|rd|th)?(?!\w)")


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


def spell_english_numbers(text: str) -> str:
    """Numbers in digits as English words: 3 -> three, 1,500 -> one thousand five hundred, 2.5 -> two point five,
    10% -> ten percent, 2nd -> second. Digits attached to letters (x2, 3d) are left, so strip_markup rejects them."""
    def say(m: re.Match) -> str:
        whole = int(m.group(1).replace(",", ""))
        if whole >= 10 ** 12:
            return m.group(0)
        if m.group(3) in ("st", "nd", "rd", "th") and not m.group(2):
            return f" {english_ordinal(whole)} "
        words = english_cardinal(whole)
        if m.group(2):
            words += " point " + " ".join(ONES[int(d)] for d in m.group(2))
        return f" {words}{' percent' if m.group(3) == '%' else ''} "
    return NUMBER.sub(say, text)


def to_mixed(text: str, source: str, lexicon: dict[str, str]) -> str | None:
    """A reference as a `mixed`-style target, or None if it cannot be one."""
    if source in SPELL_NUMBERS:
        text = spell_english_numbers(text)
    stripped = strip_markup(text, source)
    if stripped is None:
        return None
    return " ".join(lexicon.get(word, word) for word in normalise(stripped).split()).lower() or None


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


def convert(name: str, lexicon: dict[str, str], lexicon_path: Path) -> dict:
    """Write labels-mix.jsonl and labels-mix.json for one source folder. The policy follows each manifest row's
    `source`, so a slice such as iv02 (rows from indicvoices_hi) gets its dataset's policy."""
    folder = DATA_ROOT / "train" / name
    labels = _shruti_labels(folder)
    counts, policies, rows = Counter(), {}, []
    for row in _read_jsonl(folder / "manifest.jsonl"):
        if row["duration"] > MAX_SECONDS:
            counts["over_30s"] += 1
            continue
        source = row.get("source") or name
        rule = policies.setdefault(source, policy(source))
        label = (labels or {}).get(row["id"])
        target = to_mixed(row.get("text") or "", source, lexicon) if rule == "reference" else None
        if rule == "reference" and target is None:
            counts["reference_rejected"] += 1
        if rule == "construction":
            out = {"teacher": "", "language": label["language"] if label else row["language"], "by": "construction"}
        elif target is not None:
            out = {"teacher": target, "language": label["language"] if label and label["language"] else row["language"],
                   "by": "reference"}
        elif label is not None:
            out = {"teacher": label["teacher"], "language": label["language"], "by": "shruti"}
        else:
            counts["skipped_unlabelled"] += 1
            continue
        counts[out["by"]] += 1
        rows.append({"id": row["id"], **out})
    (folder / "labels-mix.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                             encoding="utf-8")
    meta = {"source": name, "rows": len(rows), "counts": dict(sorted(counts.items())), "policy": policies,
            "shruti_labels": "labels-shruti.jsonl" if labels is not None else None, "normaliser": VERSION,
            "lexicon": str(lexicon_path), "lexicon_sha256": hashlib.sha256(lexicon_path.read_bytes()).hexdigest()}
    if labels is None:
        meta["note"] = ("no labels-shruti.jsonl, so every clip is unlabelled: only converted references and "
                        "constructed targets get rows, and Shruti-only clips get none")
    (folder / "labels-mix.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.convert", description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="+", metavar="source", help="folders under $AKSHAR_DATA/train")
    parser.add_argument("--lexicon", type=Path, required=True, help="lexicon JSON to load, or to write with --learn")
    parser.add_argument("--learn", action="store_true", help="learn the lexicon from these sources first")
    args = parser.parse_args(argv)
    lexicon = write_lexicon(args.lexicon, args.sources) if args.learn else load_lexicon(args.lexicon)
    for name in args.sources:
        print(json.dumps(convert(name, lexicon, args.lexicon), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
