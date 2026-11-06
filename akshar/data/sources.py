"""Training sources, each pinned to an exact revision and checked against the licence ledger (data/LICENSES.md).

Every source here has a ledger row marked `admit`, `admit (D21)`, `admit, caveat` or `hold, train (D19)` (CLAUDE.md).
Adding one means adding its ledger row first. A model trained on a `hold, train` source is not released until a lawyer
has cleared it (D19), so every clip keeps its source name. Column names are as each dataset stores them at the pinned
revision.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    repo: str  # Hugging Face dataset repo
    revision: str  # exact commit
    shards: tuple[str, ...]  # parquet files, in the repo's order
    audio: str  # column holding {"bytes", "path"}
    text: str  # reference transcript, used for the blocklist check and kept for the record
    language: str  # "hi" or "en": the language flag Shruti is decoded with when labelling
    ledger: str  # status in data/LICENSES.md
    id: str | tuple[str, ...] | None = None  # id column, or columns joined with '_'; None means the audio file's name
    gated: bool = False


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source("rural_women_hi", "ai4bharat/Rural_Women_ASR_v2", "9e5e7ef5a463e78bc52f4bced7eb2c99d5411f72",
               tuple(f"data/Hindi-{i:05d}-of-00063.parquet" for i in range(63)),
               audio="audio_path", text="text", language="hi", ledger="admit, caveat"),
        Source("librispeech_100", "openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
               tuple(f"clean/train.100/{i:04d}.parquet" for i in range(14)),
               audio="audio", text="text", language="en", ledger="admit", id="id"),
        # IndicVoices' `text` writes English words in Devanagari; Track A trains on Shruti's transcripts instead.
        Source("indicvoices_hi", "ai4bharat/IndicVoices", "c96f9088f138cf89d419da7e8e643e1f05c00a87",
               tuple(f"hindi/train-{i:05d}-of-00082.parquet" for i in range(82)),
               audio="audio_filepath", text="text", language="hi", ledger="admit", gated=True),
        Source("kathbath_hi", "ai4bharat/Kathbath", "5b9e92849222026d9141acba4e8434fe816396bf",
               tuple(f"hindi/train-{i:05d}-of-00032.parquet" for i in range(32)),
               audio="audio_filepath", text="text", language="hi", ledger="admit, caveat", gated=True),
        # MUCS 2021 Hindi-English, training split; the same pinned copy as the MUCS test set (CC-BY-SA-4.0 at source).
        Source("mucs_train", "dianavdavidson/MUCS-Hinglish", "3c66ee127a7b0e0a746d4df1002858567135edc1",
               tuple(f"data/train-{i:05d}-of-00023.parquet" for i in range(23)),
               audio="audio", text="transcript", language="hi", ledger="hold, train (D19)", id="segment_id"),
        # AI4Bharat NPTEL (BhasaAnuvaad): lecture audio with source-language transcripts. The English part is
        # Indian-accented English; the Hindi part is Hindi lectures full of English technical terms.
        Source("nptel_en", "ai4bharat/NPTEL", "9e488c74c2285cceaf5cd1e3605e56b1a8584783",
               tuple(f"En-Indic/train-{i:05d}-of-00028.parquet" for i in range(28)),
               audio="chunked_audio_filepath", text="text", language="en", ledger="admit (D21)",
               id=("video_id", "start_time"), gated=True),
        Source("nptel_hi", "ai4bharat/NPTEL", "9e488c74c2285cceaf5cd1e3605e56b1a8584783",
               tuple(f"Indic-En/hin/train-{i:05d}-of-00009.parquet" for i in range(9)),
               audio="chunked_audio_filepath", text="text", language="hi", ledger="admit (D21)",
               id=("video_id", "start_time"), gated=True),
        # Shrutilipi: All India Radio news bulletins cut into clips, with the bulletin scripts (read off PDFs, not
        # verbatim) as text. Shruti's largest Hindi source. Clip ids are the file names, e.g.
        # Regional-Shimla-Hindi-0915-2021228114616_chunk_2.
        Source("shrutilipi_hi", "ai4bharat/Shrutilipi", "f4beb2f2cfaeae5a8fc28808f85165fec1cb19b3",
               tuple(f"hindi/train-{i:05d}-of-00196.parquet" for i in range(196)),
               audio="audio_filepath", text="text", language="hi", ledger="admit (D21)", gated=True),
    ]
}


def spread(n: int) -> list[int]:
    """0..n-1 in bit-reversed order, so stopping after k shards still samples evenly across the dataset."""
    bits = max(1, (n - 1).bit_length())
    order = [int(f"{i:0{bits}b}"[::-1], 2) for i in range(1 << bits)]
    return [i for i in order if i < n]


def get(name: str) -> Source:
    try:
        return SOURCES[name]
    except KeyError:
        raise SystemExit(f"unknown source {name!r}; known: {', '.join(SOURCES)}") from None
