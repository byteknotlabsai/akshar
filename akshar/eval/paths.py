"""Where data, prepared test sets and run outputs live. Datasets stay outside the repo (CLAUDE.md)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AKSHAR_DATA", "~/akshar-data")).expanduser()
RAW_ROOT = DATA_ROOT / "raw"  # files exactly as downloaded, one folder per test set
EVAL_ROOT = DATA_ROOT / "eval"  # prepared test sets: <set>/manifest.jsonl, <set>/audio/, <set>/source.json
RUNS_ROOT = Path(os.environ.get("AKSHAR_RUNS", REPO_ROOT / "runs"))  # git-ignored
