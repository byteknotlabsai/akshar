# Decisions

Each entry records what we decided, why, and what would make us revisit it. Add new entries at the bottom; never rewrite an old one. To reverse a decision, add a new entry that supersedes it.

## D1 (2026-10-31): Whistle is the base architecture

- **Decision:** build on Cactus Whistle (Apache-2.0; 55M parameters with 8 encoder and 8 decoder layers at width 512) and its Needle runtime.
- **Why:**
  - Shruti showed this architecture handles Hindi, English and Hinglish at 55M parameters.
  - The Needle engine already runs on CPUs, Android, WebAssembly and (per Cactus) iOS.
  - It includes streaming, word timestamps and keyword biasing.
- **Revisit if:** 2-bit can't hold accuracy (Phase 2), or the engine blocks a target platform (Phase 4).

## D2 (2026-10-31): "Lighter" means full-model 2-bit QAT, not LoRA

- **Status:** superseded by D10 (two variants, 4-bit and 2-bit).
- **Decision:** target ≤ 18 MB through 2-bit quantisation-aware training of all weights, with distillation from the full-precision model.
- **Why:**
  - Needle's local LoRA workflow exports at 4 bits, about 29 MB, which is Shruti's size.
  - Shruti's authors estimate 2-bit QAT at about 17 MB.
  - The engine already runs 2-bit files.
- **Consequence:** we extend needle's Apache-2.0 training and export code, because the public CLI does not expose full-model training or a 2-bit Whistle export.

## D3 (2026-10-31): Core model output uses Shruti's convention

- **Decision:** Hindi in Devanagari, English words in lowercase Latin, numbers as words, no punctuation, one spelling per word in training targets.
- **Why:**
  - It keeps comparisons with Shruti apples-to-apples.
  - Mixing script conventions inside training targets confuses the model.
- **Consequence:** Roman Hinglish, all-Devanagari, digits and punctuation are produced by a separate text layer (Phase 5).

## D4 (2026-10-31): Evaluation protocol is Shruti's, plus CER

- **Decision:** same test sets, normaliser, scorer and decoding settings as Shruti (see [roadmap.md](roadmap.md#evaluation-protocol)). We add CER for Hindi sets.
- **Why:** "matches Shruti" is only meaningful under identical scoring.
- **First milestone:** our harness reproduces Shruti's published numbers.

## D5 (2026-10-31): No data enters training without a ledger entry

- **Decision:** every dataset needs a row in [`../data/LICENSES.md`](../data/LICENSES.md) with a verified licence and commercial-use status before any of its audio is used for training.
- **Unresolved cases stay out:** MUCS 2021 is CC-BY-SA-4.0 upstream, and whether share-alike reaches trained weights is an open legal question.
- **Consequence:** Track A starts from Shruti's weights, which were trained on MUCS. Releasing Track A weights inherits that question, and the ledger records it.

## D6 (2026-10-31): Needle telemetry is always off

- **Decision:** set `NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1` in every environment that runs needle.
- **Why:** the Python package and binary report usage by default.

## D7 (2026-10-31): Free compute only; build on Ubuntu

- **Status:** amended by D14 (which machine runs which phase, and where data is stored).
- **Decision:**
  - No paid GPUs.
  - Resources: laptop RTX 3050 4 GB, Kaggle and Colab free tiers, and a 32 GB RTX machine from 2026-11-02.
  - Docs may be edited anywhere, but all building, training and evaluation runs on Ubuntu, because JAX's CUDA builds are Linux-only.
- **Revisit if:** the free tiers can't fit Track B's data volume.

## D8 (2026-10-31): Two tracks, Shruti first

- **Status:** amended by D10 (Track A now develops the 2-bit recipe for akshar-lite; Track B produces both variants).
- **Decision:**
  - **Track A** starts from Shruti's fp32 weights and aims for a 2-bit model with no accuracy loss.
  - **Track B** then trains akshar from Whistle on our own documented data mix, using the recipe Track A proved.
- **Why:**
  - Track A isolates the hardest question (does 2-bit cost accuracy, and does distillation win it back?) for a fraction of Track B's compute.
  - Track B gives us a model whose every training hour we chose and documented.
- **Consequence:** any released Track A model is a derivative of Shruti and must credit it.
- **Revisit if:** Track A can't get within 0.5 WER of Shruti at ≤ 18 MB.

## D9 (2026-10-31): Track A models are research-only; the commercial release comes from Track B

- **Decision:**
  - Weights produced in Track A (which start from Shruti) are for research and internal comparison only.
  - The commercially released akshar comes from Track B, trained only on `admit` and `admit, caveat` sources.
- **Why:** the licence check ([`../data/LICENSES.md`](../data/LICENSES.md)) put four of Shruti's ten training sources on hold:
  - Shrutilipi: All India Radio's policy allows only personal, non-commercial use.
  - NPTEL: possibly non-commercial.
  - MUCS: share-alike.
  - Common Voice: new Mozilla Data Collective terms.
  - Shruti's weights carry whatever those sources carry. Under D5, unresolved sources can't go into a commercial model.
- **Consequence:** Track A still runs first, because it answers the 2-bit question cheaply, but its output is not a product.
- **Revisit if:** the held sources are cleared (for example, written permission from AIR's News Services Division, or legal review of NPTEL, MUCS and the MDC terms).

## D10 (2026-10-31): Two release variants, akshar (4-bit) and akshar-lite (2-bit)

- **Decision:**
  - **akshar:** 4-bit, model file ≤ 30 MB (hard cap 40 MB). This is the full-quality model, and it must pass the release gate against Shruti.
  - **akshar-lite:** 2-bit, about 17 MB, made from akshar with quantisation-aware training and distillation.
    - It ships alongside akshar, with its accuracy published for every test set.
    - Provisional bar: average WER within 2 points of akshar, and 0 invented words on silence.
- **Why:**
  - The file budget is 30–40 MB, so 2 bits is no longer needed to meet it.
  - 2 bits may cost accuracy. Shipping both lets each app choose size or accuracy.
  - Peak RAM is about file size + 50 MB: roughly 80–90 MB for akshar and 65–70 MB for akshar-lite (estimates; Shruti measured 80 MB).
- **Supersedes:** D2.
- **Amends:** D8.
  - Track A now develops the 2-bit recipe and measures what 2 bits costs.
  - Track B produces both variants.
  - The 4-bit release never waits on the 2-bit work.
- **Revisit if:**
  - 2 bits costs nothing measurable: then akshar-lite becomes the default.
  - akshar-lite misses its bar: then ship akshar alone until it passes.

## D11 (2026-10-31): Three output styles, rendered from one mixed transcript

- **Decision:** akshar returns the same transcript in any of three styles, chosen per request, or all three at once:
  - `roman`: tumhara laptop accha lag raha hai
  - `devanagari`: तुम्हारा लैपटॉप अच्छा लग रहा है
  - `mixed`: तुम्हारा laptop अच्छा लग रहा है

  The model always writes `mixed` (D3). A small text layer that ships with both akshar and akshar-lite renders the other two.
- **Why:**
  - `mixed` is the only style that records which words were spoken in English. Both other styles can be built from it, but the reverse is ambiguous (is बस "bus" or "just"?).
  - One consistent target spends the model's 55M parameters on hearing rather than spelling. A model trained to write three styles would split that capacity. Since its extra targets would come from our own converter, it would only learn to imitate the converter anyway.
- **Consequences:**
  - Training transcripts in any style are converted to `mixed` first (Phase 3).
  - The text layer is a core deliverable that can start right after Phase 0, not a late extra (Phase 5).
  - Every style gets its own WER, alongside a script-agnostic WER (roadmap, Evaluation protocol).
- **Revisit if:** rendered `roman` or `devanagari` output is measurably worse than a model trained to write that style directly.

## D12 (2026-10-31): akshar runs in Windows and Mac desktop apps, and on phones

- **Decision:** akshar ships inside desktop apps on Windows and macOS, and on Android and iOS. Linux is needed only to train the model, never to run it.
- **What the Needle engine supports today** ([devices guide](https://cactuscompute.com/blog/needle-supported-devices)):
  - Windows x86_64 and ARM64: `needle.exe`, the static library `libneedle.a`, and the header `needle.h`.
  - macOS on Apple silicon only. **There is no build for Intel Macs.**
  - Also Android (arm64, armv7), iOS (arm64, library only), Linux, and WebAssembly (browser and Node).
- **How an app can use it** (choose per app stack):
  - Link `libneedle.a` and call the C API (`needle_load`, `needle_transcribe`, streaming calls). Smallest and fastest.
  - Run `needle --serve` beside the app and call it over localhost HTTP. Easiest for Electron, Tauri or Flutter apps.
  - Use the WebAssembly build in Electron or web apps. It is single-threaded; Shruti ran a 9.6 s clip in 0.45 s on an M-series Mac.
- **Consequences:**
  - The text layer (D11) must also run on Windows and macOS. Write it as data files plus a small portable library, not as Python-only code.
  - Shruti was never tested on Windows, so Phase 4 tests akshar there.
- **Open:**
  - Intel Mac support: this needs a build from Cactus, or the engine source if github.com/cactus-compute/needle includes it.
  - The app stack, which decides the integration path.

## D13 (2026-10-31): Evaluation harness details, settled by reproducing Shruti

- **Decision:** every model is scored by the harness in `akshar/eval/` ([eval.md](eval.md)). It fixes the details D4 left open:
  - **Normaliser:** Unicode NFC, lowercase, then every punctuation character (Unicode category P, including the danda) becomes a space. Digits and symbols stay.
  - **FLEURS references:** `raw_transcription`, not FLEURS' pre-normalised `transcription`.
  - **Lahaja references:** `text`, the clean Devanagari transcript, not `verbatim` or `normalized`.
  - **Monsoon:** `voi-oiwer` 0.1.4 on the raw transcript, reported as the corpus total.
  - **Romanised WER for Hinglish sets:** normalise, convert Devanagari to ITRANS (`indic-transliteration` 2.3.82), then lowercase.
  - **Silence:** 50 clips of 4 s digital silence and 50 of Gaussian noise (standard deviation 0.003, seed 20261031), decoded with `hi`.
  - **Pins:** `cactus-needle` 3.2.8 (engine 3.3.3) and every other package in `uv.lock`; each dataset at a commit in `akshar/eval/sets.py`.
  - **Sources:** LibriSpeech from the Hugging Face copy (openslr/librispeech_asr), not the OpenSLR tarball. MUCS from dianavdavidson/MUCS-Hinglish, the copy Shruti used. It is used only as a test set, never for training (`eval only` in the licence ledger).
- **Why:**
  - Scored this way, Shruti lands within 0.22 WER of Udgam Labs' numbers for the same engine build on all seven sets, and reproduces their romanised MUCS score exactly (34.79).
  - A space reproduces LibriSpeech exactly (5.95); deleting punctuation gives 7.47.
  - FLEURS' `transcription` column also maps chandrabindu to anusvara, for references only. That breaks D4's rule of one normaliser for both sides. With it Shruti would score 20.07 instead of 20.36.
- **Consequences:**
  - The remaining gaps come from the platform: Udgam Labs measured on Apple silicon, we run on x86, and the two October engine builds give identical transcripts here. So claims about akshar against Shruti use our own Shruti runs on the same machine, never Udgam Labs' numbers.
  - Changing the normaliser, a scorer, the engine build or a dataset revision needs a new decision, and Shruti must be re-scored with it.
  - The Phase 0 exit check passes on all seven sets.
- **Revisit if:** a new engine build changes Shruti's transcripts.

## D14 (2026-10-31): Which machine does what, and where data lives

- **Decision:**
  - The laptop (RTX 3050, 4 GB) runs Phases 1 and 2 and the text layer (Phase 5).
  - The 32 GB VRAM Ubuntu machine, reached over SSH from the laptop, runs Phase 3: pseudo-labelling, and training akshar and akshar-lite.
  - Training reads its data from the training machine's own disk, never from a network mount.
  - Google Drive (the owner's 5 TB plan) stores what we create: pseudo-labels, packed training shards, checkpoints and backups. Files go up as large packs, because Drive is slow per file and caps uploads at 750 GB a day.
  - Source datasets are downloaded from Hugging Face, at pinned revisions, on the machine that uses them. They are not copied between machines.
  - Phase 4 uses devices the owner provides: low-end and mid-range Android phones, a Windows PC and an Apple-silicon Mac.
- **Why:**
  - The model is small. Training all 55M weights, with optimiser state and a frozen teacher for distillation, takes about 1 GB of GPU memory before activations, so the 4 GB laptop GPU handles Phases 1 and 2 with small batches and gradient accumulation. The earlier plan gave Track A to the 32 GB machine without needing to.
  - Phase 3 processes thousands of hours, several times over. That would take days per run on the laptop, and its audio would not fit on the laptop's disk.
  - The internet link is 100 Mbps, so 1 TB takes about a day to move. Large data should travel once, straight from its source.
- **Amends:** D7.
- **Open:** the 32 GB machine's free disk, RAM, CPU and network speed, and whether it can run multi-day jobs unattended. The owner will confirm them.

## Superseded from the earlier ChatGPT plan

- The project name BharatSTT is now **akshar**.
- "LoRA fine-tuning first" is superseded by D2. LoRA remains a plumbing smoke test only.
- "Peak memory below 256 MB" is now ≤ 90 MB for akshar and lower for akshar-lite (D10). Shruti measured 80 MB.
- The "825-clip FLEURS download" milestone is replaced by the D4 harness milestone.
