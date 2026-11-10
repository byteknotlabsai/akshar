"""Rewrite the English words a lexicon knows into Latin inside existing label files (Shruti's or akshar's own
transcripts), token by token, writing labels-<out>.jsonl beside the input. Used for the Shrutilipi sources, whose
targets are model transcripts rather than converted references (D32).

    AKSHAR_DATA=... python scripts/apply_lexicon.py LEXICON.json OUT_NAME SOURCE [SOURCE...]
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akshar.data.convert import load_lexicon_table, fix_hindi_numbers
from akshar.eval.paths import DATA_ROOT

lexicon = load_lexicon_table(Path(sys.argv[1])); out_name = sys.argv[2]
for name in sys.argv[3:]:
    folder = DATA_ROOT / "train" / name
    src = folder / "labels-mix.jsonl"
    rows, changed, tokens = [], 0, 0
    for line in open(src, encoding="utf-8"):
        row = json.loads(line)
        words = row["teacher"].split()
        if row.get("language") == "hi":
            new = fix_hindi_numbers([lexicon.convert(w) for w in words])
        else:
            new = words
        teacher = " ".join(new)
        changed += teacher != row["teacher"]; tokens += sum(a != b for a, b in zip(words, new))
        rows.append({**row, "teacher": teacher})
    (folder / f"labels-{out_name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"source": name, "rows": len(rows), "changed_rows": changed, "changed_tokens": tokens, "out": f"labels-{out_name}.jsonl"}))
