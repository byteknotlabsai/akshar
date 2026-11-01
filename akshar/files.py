"""Small file helpers shared by the resumable writers (labelling, building, eval runs)."""
from __future__ import annotations

import os
from pathlib import Path


def repair_jsonl(path: Path) -> int:
    """Cut a half-written last line off a JSON-lines file, as a killed writer can leave. Returns bytes removed."""
    if not path.exists():
        return 0
    size = path.stat().st_size
    if size == 0:
        return 0
    with open(path, "rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return 0
        step, pos = 4096, size
        while pos > 0:  # find the last newline from the end, a block at a time
            start = max(0, pos - step)
            handle.seek(start)
            block = handle.read(pos - start)
            cut = block.rfind(b"\n")
            if cut >= 0:
                handle.truncate(start + cut + 1)
                return size - (start + cut + 1)
            pos = start
        handle.truncate(0)
        return size
