"""Whistle checkpoints, quantisation and .cact export, built on needle's Apache-2.0 code (cactus-needle 3.2.8)."""
import os

# D6: the cactus-needle package reports usage unless this is set before it is imported.
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")
