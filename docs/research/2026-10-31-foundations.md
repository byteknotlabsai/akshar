# Foundations: what Whistle, Needle and Shruti really are

Research date: 2026-10-31. Every claim below comes from the sources listed at the end. Numbers marked "Udgam Labs" were measured by Shruti's authors, not by us.

> **Update, later on 2026-10-31:** decision D10 replaced the size targets in sections 4 and 5. akshar ships at 4 bits (file ≤ 30 MB, hard cap 40 MB) alongside a 2-bit akshar-lite (about 17 MB). The facts in this note are unchanged.

## 1. Cactus Whistle (the base architecture)

- **What it is:** an on-device speech-to-text model from Cactus Compute. Licence Apache-2.0.
- **Size:** one 16.9 MB file, `whistle.cact`, quantised after training to about 2 bits per weight ("Cactus Quants at 2.125 bits per weight").
- **Architecture:** log-mel front end, convolutional stem and audio encoder. A Needle-style text decoder reads the encoder through gated cross-attention at every layer. Shruti's card gives the shape: 8 encoder layers, 8 decoder layers, width 512, about 55M parameters.
- **Decoder depth is selectable:** depths from 2 to 8 layers are each deployable (`--layers N` at build time). This gives us a free size/accuracy dial.
- **Vocabulary:** 8,192-piece SentencePiece (from Shruti's report).
- **Languages:** English, German, French, Spanish, Italian, Dutch, Polish. **No Hindi and no Devanagari.** The fine-tuning guide warns that scripts the base model has never seen "need far more data before the output is usable".
- **Input and features:** 16 kHz mono, 30 s per pass with longer audio windowed. It supports streaming, word timestamps, keyword biasing and speech embeddings (one row per 80 ms).
- **Runtime:** the Needle engine, with a C API (`needle_load`, `needle_transcribe`, `needle_stream_transcribe_process`, ...), Python, a CLI and a WebAssembly build. The same binary runs Needle (Cactus's text model) and Whistle.
- **Platforms** ([devices guide](https://cactuscompute.com/blog/needle-supported-devices), [engine repo](https://huggingface.co/Cactus-Compute/needle3)):
  - Windows x86_64 and ARM64.
  - macOS on Apple silicon only (no Intel build).
  - Linux: x86_64, ARM64, ARMv7, RISC-V and MIPS.
  - Android: arm64, armv7 and riscv64.
  - iOS, watchOS and tvOS (static library only).
  - WebAssembly, and a WASI component.
  - Native folders ship `needle.exe`, `libneedle.a` and `needle.h`. The runner has a `--serve` localhost HTTP mode.
  - The engine repo is Apache-2.0. We haven't confirmed whether the engine's C source is public.

## 2. Needle tooling (`cactus-needle`)

- Version 3.2.8, released 2026-10-19, Apache-2.0. It is a pure-Python wheel and downloads the native engine separately.
- Extras: `train`, `gpu`, `metal`, `mic`, `compare`, `test`, `mlx`. Training is JAX on CPU, CUDA or Metal. **JAX's CUDA builds are Linux-only**, so GPU training needs Ubuntu (or WSL2).
- **Telemetry is on by default.** Set `NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1`.
- **Local Whistle fine-tuning is LoRA only:**
  ```
  needle whistle finetune train.jsonl validation.jsonl test.jsonl --tokens tokens.txt --out adapter.safetensors
  needle whistle build --lora adapter.safetensors --out tuned.cact
  ```
  Documented flags: `--language`, `--tokens`, `--epochs`, `--lora-rank`, `--lora-alpha`, `--lr`, `--batch-size`, `--checkpoint <base.safetensors>`, `--checkpoint-dir`, `--out`. Defaults: 3 epochs, batch 16, lr 1e-4.
  - Data is JSONL, one clip per line: `{"audio": "clips/0001.wav", "text": "...", "language": "hi"}`. A language code outside the 7 known ones adds a new language.
  - Clips over 30 s, empty transcripts and transcripts over 320 tokens are skipped.
  - `--tokens` adds each line (character or word) as one new token with its own embedding row.
  - The test score printed after training is a greedy decode on 4-bit weights. Real numbers need the built `.cact`.
  - Example from the guide: about 10 h of FLEURS Brazilian Portuguese with `--lr 1e-2` took WER from 106.2% to 30.5%.
- **Export size:** a local build is 4-bit and about 29 MB. **This is the same size as Shruti, so LoRA cannot make akshar lighter.** Full-model training and 2-bit export are only offered through Cactus's paid hosted platform. `needle build` (for the text model) has a `--bits 2|4` option; we have not yet checked whether Whistle's build can use it.

## 3. Shruti (our reference and Track A starting point)

Udgam Labs released Shruti around 2026-10-19 (evaluation dated 2026-10-18/19).

- **Licence:** Apache-2.0 weights. Whistle is credited in its NOTICE.
- **Files:** `shruti.cact` is 29.5 MB (4-bit weights, 8-bit activations) and `shruti.safetensors` is 222 MB (fp32, config in the metadata).
- **Training method:**
  - Full fine-tune of all 55M weights, not LoRA.
  - Quantisation-aware training (QAT) in every forward pass: 4-bit weights, 8-bit activations and KV cache, straight-through estimator.
  - Two stages: Whistle weights → Hindi/Hinglish/English mix → more Indian English.
  - Silence and synthetic-noise clips with empty transcripts are included.
  - Not published: optimiser, learning rate, batch size, steps, hardware and compute.
- **Vocabulary:**
  - 714 Devanagari pieces were added to Whistle's 8,192 (characters plus common Hindi subwords, each in a word-start and a word-internal form).
  - Each new row starts as the mean of the UTF-8 byte-piece rows the old tokenizer would have used.
  - A `hi` language tag starts as the mean of the existing language-tag rows, and original token ids are kept.
- **Data:** 2,732 h. Sources: IndicVoices (hi), Vaani (hi, transcribed), Shrutilipi (hi), Kathbath (hi), Rural Women ASR v2 (hi), FLEURS (hi train), MUCS 2021 Hindi-English (train), AI4Bharat NPTEL, Common Voice 17 (Indian/South Asian English) and LibriSpeech.
  - Test clips were removed by normalised-text hash (27,710) and audio hash (17,115).
  - The authors admit the blocklist was applied unevenly, so some Svarah and Lahaja overlap checks are incomplete.
- **Output convention:**
  - Hindi in Devanagari and English words in lowercase Latin.
  - Numbers as words, no punctuation, no capitals.
  - For training targets only, nukta and chandrabindu variants were mapped to the most frequent spelling.
- **Decoding:** 5 beams with the language set, or auto-detect (about the same except for all-English Indian-accented speech, where `en` helps: 21.1 vs 25.8 WER on a Svarah subset).

### Shruti's published results (WER %, Udgam Labs)

| Model | Size | Monsoon Hindi | FLEURS Hindi | Lahaja | MUCS Hinglish | Svarah | LibriSpeech clean | Invented words, 100 silent clips |
|---|---|---|---|---|---|---|---|---|
| **Shruti** | 29.5 MB | 20.46 | 19.92 | 26.28 | 35.21 | 22.47 | 5.95 | 0 |
| Koyal-hi-120M | 481 MB | 11.64 | 9.56 | 18.70 | 57.51 | Hindi only | Hindi only | 0 |
| IndicConformer Hindi (fp32) | 482 MB | 15.31 | 11.31 | 13.18 | 64.48 | Hindi only | Hindi only | 88 |
| Vosk small Hindi + Indian English | 139 MB | 35.75 | 18.07 | 26.60 | 61.14 | 38.51 | 30.18 | 1 |
| Whisper large-v3 (fp16, subsets) | 3,087 MB | 33.29 | 28.88 | 31.19 | 79.94 | 7.39 | 2.25 | 538 |

Test-set sizes: Monsoon 753, FLEURS Hindi 418, Lahaja 6,152, MUCS 3,136, Svarah 6,656, LibriSpeech test-clean 2,620.

Scoring:
- Normaliser: Unicode NFC, lowercase, punctuation removed. Scorer: word-level edit distance.
- Monsoon (the Hindi track of the Hugging Face Open ASR leaderboard) uses the leaderboard's OIWER lattice scorer, which accepts spelling variants.
- Published leaderboard numbers for Monsoon are Qwen3-ASR-0.6B 18.13 and IndicConformer-600M 8.47.

**Phone numbers** (Samsung Galaxy F62, Exynos 9825, 2 threads):
- About 1.0 s for a 7 s sentence, including model load.
- 60 s of audio in 9.5 s.
- **80 MB peak RAM.**

**Weaknesses the authors list:**
- Hindi-only models about 16× larger make roughly half the errors.
- English trails Whisper large-v3 by a wide margin.
- Common confusions: है/हैं, मैं/में, की/के, पे/पर, वो/वह.
- Unseen names are spelt by sound.
- Noise and music were not tested.

**Their stated next steps:**
- 2-bit QAT (they estimate about 17 MB).
- More conversational Hindi under open licences, and pseudo-labels from a permissive Hindi model.
- More Indian-English conversational speech.
- Tests with noise and music.

## 4. What this means for akshar

1. **"Lighter with no compromise" is a quantisation problem first.**
   - 55M parameters at 4 bits is about 29.5 MB. At Whistle's ~2.5 bits per parameter (including overheads) it is about 17 MB.
   - Getting there needs full-model 2-bit QAT. LoRA can't do it.
   - The engine already runs 2-bit files (the shipped `whistle.cact` is one), so this is an export-and-training task, not a runtime change.
2. **Distillation is the obvious way to recover what 2 bits loses.** The fp32 model and the 2-bit student share a tokenizer, so we can match their output distributions token by token.
3. **Shruti's accuracy is not the ceiling.** On pure Hindi, 481 MB models do about twice as well. Shruti's real lead is Hinglish (MUCS 35.21 versus 57+ for every other model) and size. Better Hindi at the same size needs more conversational data and pseudo-labels, which is Track B.
4. **The evaluation must match Shruti's exactly** (same clips, normaliser, scorer, beams and language flag), or "matches Shruti" means nothing.
5. **Our licence check raises questions about four of Shruti's training sources,** although its card describes all of its data as licensed for commercial use.
   - Shrutilipi is All India Radio audio, and AIR's policy allows only personal, non-commercial use.
   - NPTEL may be non-commercial, MUCS is share-alike, and Common Voice now comes with Mozilla Data Collective terms.
   - Anything built on Shruti's weights inherits this, which is why Track A is research-only (decision D9).
   - Details: `data/LICENSES.md`.

## 5. Corrections to the earlier (ChatGPT) plan

| Earlier plan said | What we found | What changes |
|---|---|---|
| Fine-tune Whistle with LoRA, merge, export | That is the only local workflow, and it exports at 4-bit, about 29 MB (Shruti's size). Shruti trained all weights with QAT. | LoRA is only a plumbing test. Lighter needs full-model 2-bit QAT. |
| `needle whistle finetune/build` are "planned" commands | They exist in `cactus-needle[train]` 3.2.8 as written. | Usable as documented. |
| Model footprint target 20–35 MB | Shruti is already 29.5 MB. | Target ≤ 18 MB. |
| Peak memory ideally below 256 MB | Shruti peaks at 80 MB on an Android phone. | Target ≤ 80 MB. |
| Our token script is not Shruti's 714-piece vocabulary | Shruti's vocabulary method is published (see section 3). | Track B can reproduce it. |
| MUCS Hinglish on legal hold | Shruti trained on MUCS through an HF copy labelled CC-BY-4.0. The original corpus is CC-BY-SA-4.0. | Still our decision (see `data/LICENSES.md`). Track A inherits it through Shruti's weights. |
| First milestone: download 825 FLEURS clips | Only a plumbing test. | First milestone: our eval harness reproduces Shruti's published numbers. |
| Shruti is only a benchmark | Its weights are Apache-2.0. | Track A builds on them, with credit. |
| Hardware: RTX 3050 4 GB, Ubuntu 24.04 | Confirmed 4 GB (the Windows side has no Python or WSL; Ubuntu is a dual-boot). | All building happens on Ubuntu. A 32 GB RTX machine is available from 2026-11-02. |

## 6. Open questions to answer on Ubuntu

1. Does `cactus-needle`'s training code let us unfreeze all weights, or do we add that? Shruti's report says they used needle's Whistle code for QAT and embedding initialisation.
2. Can we export a Whistle checkpoint at 2 bits with the open-source writer (`--bits 2|4` exists for `needle build`)? Does fake-quant during training match the engine's 2-bit dequantisation exactly?
3. Does `shruti.safetensors` load into needle's training code and reproduce `shruti.cact` outputs?
4. ~~Where do we get Monsoon and the OIWER scorer?~~ **Resolved:**
   - The public split is [VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN): CC-BY-4.0, 753 clips, the same count Shruti reports.
   - The scorer is `voi-oiwer` (MIT). References are spelling lattices such as `इम्प्रूव | improve` and `एक | 1`.
   - See `data/LICENSES.md`.
5. ~~What licence does Koyal-hi-120M carry?~~ **Resolved:** `adalat-ai/koyal-hi-120m-1.0` is CC-BY-4.0 and gated (you share your HF details).
   - It is a 120M Conformer hybrid TDT-CTC model with a 512-piece tokenizer.
   - It trained on about 2,400 h of public Hindi (IndicVoices, Shrutilipi, Kathbath, OpenSLR 118, Common Voice, FLEURS), with an encoder warm-started from AI4Bharat's MIT-licensed MahaDhwani.
   - It writes punctuation and digits, so its pseudo-labels must be converted to our convention (numbers as words, no punctuation).
   - It is a usable Hindi teacher with attribution.
6. ~~IndicConformer licence conflict.~~ **Resolved:** the official `ai4bharat/indic-conformer-600m-multilingual` card says MIT, and the model is gated (you share your HF details).
   - It is a hybrid CTC + RNNT model covering 22 languages; Hindi WER is 13.2 on the Vaani benchmark.
   - The card recommends the newer Indic-Transcribe model, whose licence we haven't checked.
   - Neither card says how English words are written, so test that before using either as a Hinglish teacher.

## Sources

- Whistle model card: https://huggingface.co/Cactus-Compute/whistle
- Needle repository: https://github.com/cactus-compute/needle
- Whistle fine-tuning guide: https://cactuscompute.com/blog/finetuning-whistle
- Needle reference for assistants: https://raw.githubusercontent.com/cactus-compute/needle/main/llms.txt
- `cactus-needle` on PyPI: https://pypi.org/project/cactus-needle/
- Shruti model card: https://huggingface.co/udgamlabs/Shruti
- Shruti technical report: https://udgamlabs.com/research/shruti
- Shruti repository: https://github.com/udgamlabs/shruti
- IndicConformer 600M model card: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
- IndicConformer repository: https://github.com/AI4Bharat/IndicConformerASR
- Koyal-hi-120M model card: https://huggingface.co/adalat-ai/koyal-hi-120m-1.0
