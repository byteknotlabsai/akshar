# Evaluation harness

Phase 0 of the [roadmap](roadmap.md). Every akshar or Shruti number we report comes from this code, in [`akshar/eval/`](../akshar/eval/). It implements the [evaluation protocol](roadmap.md#evaluation-protocol) (decision D4). Decision D13 fixes the details below.

**Status (2026-10-31):** Shruti, scored by this harness, lands within 0.22 WER of Udgam Labs' numbers for the same engine build on all seven test sets. The Phase 0 exit check passes.

## Running it

On Ubuntu, after [setup-ubuntu.md](setup-ubuntu.md), from the repo root:

```bash
uv run python -m akshar.eval prepare all
```

```bash
uv run python -m akshar.eval run --model models/shruti/shruti.cact --set fleurs_hi
```

```bash
uv run python -m akshar.eval report
```

| Command | What it does |
|---|---|
| `prepare <set>...` or `prepare all` | Downloads each set at its pinned revision into `~/akshar-data/raw/<set>/`. Then writes `~/akshar-data/eval/<set>/`: `manifest.jsonl` (clips in the source's order), `audio/`, and `source.json` (revision, file hashes, clip count). |
| `run --model <file.cact> --set <set>` | Decodes every clip in the needle engine with the protocol's settings. Writes `runs/<set>/<run>/`: `hyps.jsonl` (each clip's transcript), `run.json` (model and engine hashes, settings, machine, memory) and `scores.json`. |
| `score <run folder>... [--first N]` | Re-scores saved transcripts. `--first N` scores only the first N clips, as in Udgam Labs' subsets. |
| `report` | Every complete run in one table, beside Udgam Labs' number for the same engine build. |
| `blocklist` | Text and audio hashes of every test clip, for removing them from training data. |

More `run` options:
- `--language auto` lets the engine detect the language.
- `--limit N` decodes only the first N clips.
- `--out <run folder>` resumes a run that stopped.
- `NEEDLE3_LIB_PATH=<libneedle.so>` runs a different engine build; `run.json` records which one ran.

The harness sets `NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1` before it imports needle (D6). `AKSHAR_DATA` and `AKSHAR_RUNS` move the data and run folders.

**Lahaja and Svarah are gated.** Accept their terms on Hugging Face while logged in, run `hf auth login` with a token from that account, then `prepare lahaja svarah`. The token must be a classic "Read" token, or a fine-grained one with "Read access to contents of all public gated repos you can access" ticked ([setup-ubuntu.md](setup-ubuntu.md), step 11).

## The protocol as implemented

| Set | Source, pinned revision | Clips | Language flag | Scores |
|---|---|---|---|---|
| Monsoon Hindi | [VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN) @ a4b7375 | 753 | hi | OIWER |
| FLEURS Hindi | [google/fleurs](https://huggingface.co/datasets/google/fleurs) @ 70bb2e8, `hi_in` test | 418 | hi | WER, CER |
| Lahaja | [ai4bharat/Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja) @ d4ffd2e (gated) | 6,152 | hi | WER, CER |
| MUCS Hinglish | [dianavdavidson/MUCS-Hinglish](https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish) @ 3c66ee1, test | 3,136 | hi | WER, CER, romanised WER |
| Svarah | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) @ ebbf777 (gated) | 6,656 | en | WER |
| LibriSpeech test-clean | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) @ 71cacbf | 2,620 | en | WER |
| Silence | generated | 100 | hi | words written |

- **Engine:** `cactus-needle` 3.2.8, pinned in `pyproject.toml`. It loads engine 3.3.3 (needle3 @ d6c7d6a, 19 October 2026). Decoding uses the engine's defaults: 5 beams. The C API has no beam or thread setting.
- **Audio:** decoded with soundfile, channels averaged, and resampled to 16 kHz with soxr at HQ quality when needed. Every current set is already 16 kHz mono.
- **Normaliser** (`normalise.py`): Unicode NFC, lowercase, then every punctuation character becomes a space. Punctuation means Unicode category P, which includes the danda (।). Digits, symbols and Devanagari signs stay. The same function runs on references and hypotheses.
- **References:** each set's own transcript.
  - FLEURS: `raw_transcription`, not FLEURS' pre-normalised `transcription` column (see below).
  - Lahaja: `text`, the clean Devanagari transcript. Its `verbatim` column adds disfluencies and noise tags ("uhh", "bird squawk"), and its `normalized` column repeats English words a second time in Latin script.
  - Lahaja clip ids are each clip's own file name, because `fname` names a whole recording that is cut into many clips.
- **WER:** word-level edit distance, summed over the set and divided by the number of reference words.
- **CER:** the same over Unicode code points, counting the spaces between words.
- **Monsoon:** `voi-oiwer` 0.1.4, called as the dataset card shows: the raw transcript against the clip's lattice, language "hindi". We report the corpus total, all errors over all reference words. Monsoon has no plain WER or CER, because its references are lattices of accepted spellings.
- **Romanised WER** (MUCS): both sides normalised, Devanagari converted to ITRANS with `indic-transliteration` 2.3.82, then lowercased. Latin-script words pass through. This is a neutral converter for the protocol's script-agnostic score, not akshar's `roman` style.
- **Silence:** 50 clips of 4 s digital silence and 50 of Gaussian noise with standard deviation 0.003 (seed 20261031), decoded with the `hi` flag. The score is the number of words written.
- **Clip order:** the source's order, so "the first N clips" are the same clips as in Udgam Labs' subsets.
- **Speed and memory:** each run records decode time per clip, the Python process's peak memory, and how much loading the model added. Real-time factor excludes audio loading. These are laptop figures; phones are Phase 4.

## Shruti through our harness

Settings for every number in this section:
- Model `shruti.cact` (SHA-256 b189e954…).
- Linux x86_64, Intel i5-12500H.
- Normaliser `nfc-lower-punct2space-v1`.
- 5 beams, language flag per set as in the table above.

Udgam Labs measured on an Apple-silicon Mac, with two engine builds: 18 October (their main table) and 19 October ("newest build", their report's section 7.5).

| Set | Ours | Udgam Labs, 19 Oct build | Ours minus theirs | Udgam Labs, 18 Oct build |
|---|---|---|---|---|
| Monsoon Hindi (OIWER) | 20.63 | 20.46 | +0.17 | 20.46 |
| FLEURS Hindi | 20.36 | 20.14 | +0.22 | 19.92 |
| Lahaja | 26.31 | 26.33 | −0.02 | 26.28 |
| MUCS Hinglish | 35.17 | 35.22 | −0.05 | 35.21 |
| Svarah | 22.47 | 22.47 | 0.00 | 22.47 |
| LibriSpeech test-clean | 5.95 | 6.00 | −0.05 | 5.95 |
| Silence (words written) | 0 | 0 | 0 | 0 |

Other scores from the same runs, plus short runs with the language detected or forced:

| Set | Measure | Ours | Udgam Labs |
|---|---|---|---|
| FLEURS Hindi | CER | 10.74 | not published |
| Lahaja | CER | 20.80 | not published |
| MUCS Hinglish | CER | 28.52 | not published |
| MUCS Hinglish | romanised WER (ITRANS) | 34.79 | 34.79 |
| Monsoon Hindi | OIWER, language detected | 20.65 (hi 751, en 2) | 20.47 (hi 751, en 2) |
| Lahaja, first 300 clips | WER, language detected | 22.85 (hi 299, en 1) | 22.89 (hi 298, en 2) |
| Svarah, first 400 clips | WER, language detected | 25.75 (hi 257, en 142, none 1) | 25.75 (hi 257, en 142, none 1) |
| Svarah, first 400 clips | WER, language hi | 27.50 | 27.52 |
| LibriSpeech, first 400 clips | WER, language detected | 5.99 (en 400) | 6.04 (en 400) |

Speed on this laptop: 42 to 66 times faster than real time, depending on the set. Loading the model added 34 MB to the process.

### How the details were settled

- **Punctuation becomes a space, not nothing.** On LibriSpeech, space gives 5.95, exactly Udgam Labs' 18 October number. Deleting punctuation gives 7.47. FLEURS moves the same way (20.36 against 21.08); MUCS barely moves.
- **Clip order matches Udgam Labs'.** Their subset scores, set against ours from the same full runs:

  | Subset | Ours | Udgam Labs |
  |---|---|---|
  | LibriSpeech, first 300 | 5.71 | 5.76 |
  | LibriSpeech, first 400 | 5.99 | 6.04 |
  | LibriSpeech, first 1,000 | 6.30 | 6.36 |
  | Lahaja, first 300 | 22.87 | 22.87 |
  | MUCS, first 400 | 49.90 | 49.89 |
  | MUCS, first 600 | 45.57 | 45.3 |
  | Svarah, first 300 | 21.45 | 21.35 |
  | Svarah, first 400 | 21.15 | 21.13 |
  | Svarah, first 1,500 | 21.37 | 21.35 |

- **Lahaja uses `text`.** Against it Shruti scores 26.31. Against `verbatim` it scores 26.91, and against `normalized` 26.55.
- **FLEURS uses `raw_transcription`.** FLEURS' `transcription` column also turns chandrabindu into anusvara (बाँधने → बांधने) in 80 clips, which is how Shruti spells. With that column Shruti scores 20.07 instead of 20.36. But it normalises references with a function the hypotheses never get, which breaks the protocol's one rule for the normaliser.
- **The engine's output is nearly identical.** Udgam Labs' report prints Shruti's transcripts of the first Monsoon clips. Four of the five match ours word for word. The fifth reads "बैंग काम" in theirs and "bank काम" in ours.
- **The build is not the difference.** On this laptop, engine 3.3.2 (their 18 October build) and 3.3.3 give identical transcripts on all 6,927 clips of Monsoon, FLEURS, MUCS and LibriSpeech. Udgam Labs saw the two builds differ on their Mac (FLEURS 19.92 against 20.14). The gaps that remain are therefore x86 against Apple silicon, and they run in both directions.

### Phase 0 exit check

The check: `shruti.cact` scored by our harness lands within 0.3 WER of Udgam Labs' numbers on every set we can get.

- **Passed** on all seven sets, against Udgam Labs' numbers for the same engine build. The largest gap is FLEURS at +0.22.
- FLEURS is 0.44 above their 18 October number (19.92). Our x86 engine gives the same transcripts with either build, so no setting of ours reaches 19.92.

Because of these platform gaps, claims about akshar against Shruti always use our own Shruti runs, on the same machine, never Udgam Labs' numbers (CLAUDE.md).

## Contamination blocklist

`python -m akshar.eval blocklist` writes `~/akshar-data/eval/blocklist.jsonl`. For each test clip it holds two hashes:
- SHA-256 of the normalised reference text. For FLEURS and Lahaja, their other reference columns are hashed too.
- SHA-256 of the audio as 16 kHz mono 16-bit PCM.

On 2026-10-31 it covered all 19,735 clips of the six recorded test sets: 24,825 distinct text hashes and 19,735 audio hashes. Phase 3 data preparation must drop every training clip that matches either hash. Rebuild it whenever a test set is added.

Two limits for Phase 3:
- An audio hash only matches identical audio, so a re-encoded or resampled copy gets through. The text hash is the main guard.
- The hashed Monsoon text is the first spelling of each span, which is Devanagari, and Lahaja's `text` is Devanagari only. Hinglish training labels must therefore also be checked in a Devanagari-only form, as Shruti's authors did.

## Not done yet

- Per-style WER: needs the three-style reference set from Phase 5.
- Device speed and memory: Phase 4. The memory figures here include the Python process.
