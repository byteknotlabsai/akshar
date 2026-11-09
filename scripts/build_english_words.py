"""Build data/english-words (D32): merge the category lists written by Claude Opus agents (raw/*.tsv) into one
curated table of English words that akshar writes in Latin, drop what the blocklist, the agents' AMBIGUOUS notes,
fixes.tsv and the annotators' glosses say is a Hindi word, and build the converter lexicon, in which the entries
learned from Shruti (lexicon-iv) win and are the only ones found under folded spellings.

    python scripts/build_english_words.py data/english-words/raw data/english-words ~/akshar-data/server/night3/lexicon-iv.json

Needs raw/gloss-stats.json and raw/unglossed-counts.json beside the lists: how often Vaani's and Rural Women's
annotators glossed each Devanagari token as English and left it unglossed (built from their manifests on 2026-11-09).
Writes words.tsv (kept), review.tsv (dropped, with reasons), blocklist.tsv and lexicon-v4.json.
"""
import csv, json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akshar.data.convert import fold_devanagari
from akshar.eval.normalise import normalise

agents, out, lex_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
stats = json.loads((agents / "gloss-stats.json").read_text(encoding="utf-8"))
unglossed = json.loads((agents / "unglossed-counts.json").read_text(encoding="utf-8"))
learned_full = json.loads(lex_path.read_text(encoding="utf-8"))
learned = {w: e[0] for w, e in learned_full["words"].items()}
LISTS = ("tech", "objects", "institutions", "brands", "places_world", "places_india", "common")
INFLECTED = ("ों", "ें", "ियों", "ियां", "ियाँ", "याँ", "यां")
ABBREV_NOTE = re.compile(r"AMBIGUOUS:\s*abbreviation spoken letter by letter\.?", re.I)

def nfc(s): return unicodedata.normalize("NFC", s.strip())
def devanagari_only(tok): return all("ऀ" <= ch <= "ॿ" or ch in "‌‍" for ch in tok)

block = {}
for row in csv.DictReader(open(agents / "blocklist.tsv", encoding="utf-8"), delimiter="\t"):
    for tok in nfc(row["devanagari"]).split("|"):
        tok = normalise(tok).strip()
        if tok: block[tok] = f'{row["group"]}: {row["reason"]}'
block_folded = {fold_devanagari(t) for t in block}
excl, over = {}, {}
for line in open(out / "fixes.tsv", encoding="utf-8"):
    if line.startswith("#") or not line.strip(): continue
    kind, dev, english, reason = (line.rstrip("\n").split("\t") + ["", ""])[:4]
    (excl if kind == "exclude" else over)[normalise(dev).strip()] = (english.strip(), reason)

rows, review, seen = [], [], set()
for name in LISTS:
    for row in csv.DictReader(open(agents / f"{name}.tsv", encoding="utf-8"), delimiter="\t"):
        english = " ".join(nfc(row.get("english") or "").lower().replace(".", "").split())
        cat = nfc(row.get("category") or name); note = nfc(row.get("note") or "")
        raw = row.get("devanagari") or ""
        if not english or not english.isascii() or not english.replace(" ", "").replace("-", "").isalnum():
            review.append((english, raw, cat, "english not plain ascii letters")); continue
        if "HINDI_NAME" in note.upper():  # the Hindi name is a different word (चीन): transcribe, never translate
            review.append((english, raw, cat, f"agent: {note}")); continue
        if "AMBIGUOUS" in note.upper() and not ABBREV_NOTE.fullmatch(note):
            review.append((english, raw, cat, f"agent: {note}")); continue
        spellings = [normalise(d).strip() for d in nfc(raw).split("|")]
        spellings = [d for d in spellings if d and devanagari_only(d.replace(" ", ""))]
        for dev in spellings:
            if " " in dev:
                review.append((english, dev, cat, "spaced spelling: not a token")); continue
            if dev in excl:
                review.append((english, dev, cat, f"fixes: {excl[dev][1]}")); continue
            if dev in block or fold_devanagari(dev) in block_folded:
                review.append((english, dev, cat, f"blocklist: {block.get(dev, 'folded form blocked')}")); continue
            if len(spellings) > 1 and (dev.endswith(INFLECTED) or (dev.endswith("े") and dev[:-1] + "ा" in spellings)):
                review.append((english, dev, cat, "inflected Hindi form; the base spelling is listed")); continue
            target = over[dev][0] if dev in over else english
            g = stats.get(dev); un = unglossed.get(dev, 0); gl = g["glossed"] if g else 0
            if not cat.startswith("place") and un >= 50 and gl / max(gl + un, 1) < 0.15:
                review.append((target, dev, cat, f"annotators leave it in Devanagari: glossed {gl}, unglossed {un}"
                               + (f"; glosses {g['latin'][:2]}" if g else ""))); continue
            if dev not in over and g and g["latin"] and g["latin"][0][1] >= 10 \
                    and g["latin"][0][0].replace(" ", "") != target.replace(" ", ""):
                review.append((target, dev, cat, f"annotators gloss it as {g['latin'][0][0]} ({g['latin'][0][1]}x)")); continue
            if (dev, target) in seen: continue
            seen.add((dev, target)); rows.append((target, dev, cat, note, gl, un))

by_dev = defaultdict(set)
for english, dev, *_ in rows: by_dev[dev].add(english)
kept = []
for english, dev, cat, note, gl, un in rows:
    if len(by_dev[dev]) > 1 and dev not in over:
        review.append((english, dev, cat, f"conflict: {dev} also -> {sorted(by_dev[dev] - {english})}")); continue
    if dev in over and english != over[dev][0]: continue
    if dev in learned and learned[dev] != english:
        review.append((english, dev, cat, f"learned lexicon writes {learned[dev]}; kept the learned spelling")); continue
    kept.append((english, dev, cat, note, gl, un))

with open(out / "words.tsv", "w", encoding="utf-8") as h:
    h.write("english\tdevanagari\tcategory\tnote\tglossed\tunglossed\n")
    for r in sorted(kept, key=lambda r: (r[2], r[0], r[1])): h.write("\t".join(str(x) for x in r) + "\n")
with open(out / "review.tsv", "w", encoding="utf-8") as h:
    h.write("english\tdevanagari\tcategory\treason\n")
    for r in review: h.write("\t".join(str(x) for x in r) + "\n")
with open(out / "blocklist.tsv", "w", encoding="utf-8") as h:
    h.write("devanagari\treason\n")
    for tok, why in sorted(block.items()): h.write(f"{tok}\t{why}\n")
    for tok, (_, why) in sorted(excl.items()): h.write(f"{tok}\tFIXES: {why}\n")
new = {dev: [english, 0, 0] for english, dev, *_ in kept if dev not in learned}
lexicon = {"sources": learned_full["sources"] + ["curated english-words (D32)"], "min_count": 5, "min_share": 0.8,
           "words": {**learned_full["words"], **new},
           "blocked": sorted((set(block) | set(excl)) - set(learned))}  # never converted (D32); learned entries stay
(out / "lexicon-v4.json").write_text(json.dumps(lexicon, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
cats = Counter(r[2] for r in kept)
print("kept", len(kept), "spellings for", len({r[0] for r in kept}), "English words;", dict(sorted(cats.items())))
print("review", len(review), "; blocklist", len(block), "+", len(excl), "fixes")
print("lexicon-v4:", len(lexicon["words"]), "entries (", len(learned), "learned +", len(new), "curated )")
