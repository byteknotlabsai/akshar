"""Audio decoding to 16 kHz mono float32, done the same way for every model and every set."""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def load(path: str | Path) -> np.ndarray:
    """16 kHz mono float32 samples in [-1, 1].

    Channels are averaged and other rates are resampled with soxr at HQ quality, as the needle package does for WAV
    files (needle/agent/whistle.py, _read_wav). Shruti's report says its website resampler matches soxr.
    """
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1, dtype=np.float32)
    if rate != SAMPLE_RATE and len(mono):
        import soxr

        mono = soxr.resample(mono, rate, SAMPLE_RATE, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def audio_sha256(samples: np.ndarray) -> str:
    """Hash of the clip as 16-bit PCM at 16 kHz mono, for the contamination blocklist.

    Quantising to 16 bits first makes the hash independent of how the file stored its samples (int16 WAV, FLAC or
    float), so the same recording matches wherever it turns up.
    """
    pcm = np.clip(np.round(samples.astype(np.float64) * 32768.0), -32768, 32767).astype("<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def sniff_extension(blob: bytes) -> str:
    """File extension for encoded audio bytes, from their magic number."""
    if blob[:4] == b"RIFF":
        return ".wav"
    if blob[:4] == b"fLaC":
        return ".flac"
    if blob[:4] == b"OggS":
        return ".ogg"
    if blob[:3] == b"ID3" or blob[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    raise ValueError(f"unknown audio format, first bytes {blob[:8]!r}")
