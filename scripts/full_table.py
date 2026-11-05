"""Full-test table: per set, each named run beside Shruti's complete protocol run scored on the same machine.
usage: fulltable.py RUNS_ROOT SHRUTI_NAME|auto name..."""
import json, sys
from pathlib import Path
root, shruti, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
sets = {"monsoon": "hi", "fleurs_hi": "hi", "lahaja": "hi", "mucs": "hi", "svarah": "en", "librispeech_clean": "en", "silence": "hi"}
def load(d):
    p = d / "scores.json"
    if not p.exists(): return None
    s = json.loads(p.read_text())
    run = json.loads((d / "run.json").read_text())
    return s if s["complete"] and not run.get("limit") else None
def shruti_run(s, lang):
    if shruti != "auto":
        return load(root / s / shruti)
    for d in sorted((root / s).glob("shruti_*")):
        r = load(d)
        eng = json.loads((d / "run.json").read_text())["engine"]
        if r and r["language"] == lang and (eng.get("library_sha256") or "").startswith("87476ff70e80"):
            return r
    return None
rows = [("Shruti 4-bit", {s: shruti_run(s, l) for s, l in sets.items()})] + \
       [(n, {s: load(root / s / n) for s in sets}) for n in names]
print("| Model | " + " | ".join(sets) + " | Mean WER |"); print("|---|" + "---|" * (len(sets) + 1))
for label, res in rows:
    cells, wers = [], []
    for s in sets:
        r = res[s]
        if r is None: cells.append("–"); continue
        v = r["headline"]["value"]; cells.append(str(int(v)) if s == "silence" else f"{v:.2f}")
        if s != "silence": wers.append(v)
    print(f"| {label} | " + " | ".join(cells) + f" | {sum(wers)/len(wers):.2f} |" if len(wers) == 6 else f"| {label} | " + " | ".join(cells) + " | – |")
