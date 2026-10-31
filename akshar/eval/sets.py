"""The test sets of the evaluation protocol (docs/roadmap.md), each pinned to an exact source revision.

Clip counts and Shruti's numbers are Udgam Labs' (technical report, section 6.1 and 7.5). Shruti's numbers are WER %
(OIWER % for Monsoon, words written for silence), measured with the needle engine at 5 beams with the language set.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    repo: str  # Hugging Face dataset repo
    revision: str  # exact commit; never a branch name
    files: tuple[str, ...]


@dataclass(frozen=True)
class TestSet:
    name: str
    title: str
    language: str  # decoding language flag in the protocol: "hi" for Hindi and Hinglish sets, "en" for English sets
    scorer: str  # "wer", "oiwer" (Monsoon's lattice scorer) or "silence" (count invented words)
    clips: int  # clip count Udgam Labs reports
    cer: bool  # report CER too (Hindi sets)
    source: Source | None  # None: generated locally
    gated: bool = False  # needs `hf auth login` and accepting the dataset's terms on its page
    romanised: bool = False  # Hinglish set: also report script-agnostic (romanised) WER
    shruti_oct18: float | None = None  # Udgam Labs, engine build of 18 October 2026 (their main table)
    shruti_newest: float | None = None  # Udgam Labs, newest engine build in their report (section 7.5)


SETS: dict[str, TestSet] = {
    s.name: s
    for s in [
        # No CER for Monsoon: its references are spelling lattices, and no lattice-aware CER exists.
        TestSet("monsoon", "Monsoon Hindi (public split)", "hi", "oiwer", 753, False,
                Source("VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN", "a4b7375e9adea16c16100295976cdb0aabd9fb6d",
                       ("data/test-00000-of-00001.parquet",)),
                shruti_oct18=20.46, shruti_newest=20.46),
        TestSet("fleurs_hi", "FLEURS Hindi (test)", "hi", "wer", 418, True,
                Source("google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
                       ("data/hi_in/test.tsv", "data/hi_in/audio/test.tar.gz")),
                shruti_oct18=19.92, shruti_newest=20.14),
        TestSet("lahaja", "Lahaja (Hindi accents)", "hi", "wer", 6152, True,
                Source("ai4bharat/Lahaja", "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7",
                       tuple(f"data/test-0000{i}-of-00004.parquet" for i in range(4))),
                gated=True, shruti_oct18=26.28, shruti_newest=26.33),
        TestSet("mucs", "MUCS 2021 Hinglish (test)", "hi", "wer", 3136, True,
                Source("dianavdavidson/MUCS-Hinglish", "3c66ee127a7b0e0a746d4df1002858567135edc1",
                       ("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet")),
                romanised=True, shruti_oct18=35.21, shruti_newest=35.22),
        TestSet("svarah", "Svarah (Indian English)", "en", "wer", 6656, False,
                Source("ai4bharat/Svarah", "ebbf7777fe771490696a3f7b007097606fa8c924",
                       tuple(f"data/test-0000{i}-of-00003.parquet" for i in range(3))),
                gated=True, shruti_oct18=22.47, shruti_newest=22.47),
        TestSet("librispeech_clean", "LibriSpeech test-clean", "en", "wer", 2620, False,
                Source("openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
                       ("clean/test/0000.parquet",)),
                shruti_oct18=5.95, shruti_newest=6.00),
        TestSet("silence", "Silence (50 digital silence, 50 faint noise)", "hi", "silence", 100, False, None,
                shruti_oct18=0, shruti_newest=0),
    ]
}


def get(name: str) -> TestSet:
    try:
        return SETS[name]
    except KeyError:
        raise SystemExit(f"unknown test set {name!r}; known: {', '.join(SETS)}") from None
