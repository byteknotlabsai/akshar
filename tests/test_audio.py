import numpy as np
import soundfile as sf

from akshar.eval.audio import audio_sha256, load


def test_hash_ignores_storage_format(tmp_path):
    rng = np.random.default_rng(0)
    pcm = rng.integers(-20000, 20000, 16000, dtype=np.int16)
    sf.write(tmp_path / "a.wav", pcm, 16000, subtype="PCM_16")
    sf.write(tmp_path / "a.flac", pcm, 16000)
    sf.write(tmp_path / "a_float.wav", pcm.astype(np.float32) / 32768, 16000, subtype="FLOAT")
    hashes = {audio_sha256(load(tmp_path / name)) for name in ("a.wav", "a.flac", "a_float.wav")}
    assert len(hashes) == 1


def test_stereo_is_averaged_and_resampled(tmp_path):
    t = np.arange(8000) / 8000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(tmp_path / "s.wav", np.stack([tone, tone], axis=1), 8000, subtype="FLOAT")
    samples = load(tmp_path / "s.wav")
    assert samples.dtype == np.float32 and abs(len(samples) - 16000) <= 1
