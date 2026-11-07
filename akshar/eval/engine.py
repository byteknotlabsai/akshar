"""The needle engine library in use and the machine it runs on.

Apart from akshar.eval.run so that the page server (akshar.ui), and the deploy package built from it, can name the
engine without the harness's data-preparation code.
"""
from __future__ import annotations

import hashlib
import os
import platform

# D6: the cactus-needle package reports usage unless this is set before it is imported.
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

# Engine libraries we have run, by SHA-256 (Linux x86_64). Udgam Labs' main table used the build of 18 October 2026,
# and their "newest build" numbers came from the build of 19 October (technical report, section 7.5).
KNOWN_ENGINES = {
    "87476ff70e80334d7c766dbf236b72f42876072df22014c6ec8a9b087b82b260": "3.3.3, needle3@d6c7d6a (2026-10-19)",
    "0c8f3970fab6ca96f59ba989377c7e61def64ee0e3b3970d7a06cdf93271d454": "3.3.2, needle3@3391a10 (2026-10-18)",
}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def engine_info() -> dict:
    """The engine library actually loaded. NEEDLE3_LIB_PATH overrides the build the package pins."""
    import needle
    from needle import _library_path
    from needle.agent import fetch

    library = _library_path(3)
    digest = sha256(library)
    return {"package": needle.__version__, "pinned_engine": fetch.engine_version(3),
            "override": os.environ.get(fetch.lib_path_env(3)), "library_sha256": digest,
            "build": KNOWN_ENGINES.get(digest, "unknown")}
