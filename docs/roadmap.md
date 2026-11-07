# Roadmap

A phase is done when its exit check passes, not when its task list is ticked. Facts behind this plan: [research/2026-10-31-foundations.md](research/2026-10-31-foundations.md). Decisions: [decisions.md](decisions.md).

## Phase 0: Eval harness (first week on Ubuntu)

**Status (2026-10-31):** done. The exit check passes on all seven test sets. Details and numbers: [eval.md](eval.md).

Every later claim ("beats Shruti", "2 bits costs this much") is a number from this harness, so it comes first.

- Set up Ubuntu ([setup-ubuntu.md](setup-ubuntu.md)).
- Build the normaliser and scorer to match Shruti's (see [Evaluation protocol](#evaluation-protocol)).
- Fetch the test sets: FLEURS Hindi test, LibriSpeech test-clean, Svarah, Lahaja, the MUCS Hinglish test split, Monsoon's public hi-IN split (753 clips, CC-BY-4.0) and 100 silent clips.
- Build a runner that decodes any `.cact` through the Needle engine and logs WER, CER, RTF and peak RAM per run.
- **Exit:** `shruti.cact` scored by our harness lands within 0.3 WER of Udgam Labs' numbers on every set we can get. Start with FLEURS Hindi (19.92) and LibriSpeech (5.95).

## Phase 1: Prove the toolchain end to end

**Status (2026-10-31):** done. The exit check passes; details and numbers in [toolchain.md](toolchain.md).

- Read `cactus-needle`'s training and export code (Apache-2.0). Find:
  - where LoRA attaches;
  - the 4-bit fake-quant used in training;
  - the `.cact` writer and its 2-bit path.
- Smoke test with the stock CLI: LoRA on a few hours of FLEURS Hindi train, then `needle whistle build`, then score with the harness. This proves the plumbing only, so ignore the WER.
- Load `shruti.safetensors` (fp32) into the training code and match `shruti.cact`'s transcripts at 4 bits.
- **Exit:** we can
  1. train all weights, not just adapters;
  2. fake-quantise at 4 and 2 bits exactly the way the engine dequantises;
  3. export 4-bit and 2-bit `.cact` files that the engine runs and the harness scores.

## Phase 2: Track A, the 2-bit recipe (for akshar-lite)

**Status (2026-11-05):** exit check met. On the full test sets, the best 17.1 MB model trails 4-bit Shruti by 2.3 points of mean WER. akshar-lite's 20.4 MB recipe (D24) trails by 1.4, and by 1.1 once NPTEL joins the mix. Method, data and results in [two-bit.md](two-bit.md). Decisions D16 and D22 amend the Method and Data lines below.

- **Start:** Shruti's fp32 weights (Apache-2.0, credited in NOTICE). The output is research-only (decision D9), because Shruti was trained on sources our licence gate holds.
- **Method:**
  - 2-bit QAT on all weights.
  - Distillation from frozen fp32 Shruti (token-level KL on decoder outputs, same tokenizer) plus the normal transcript loss.
  - Try mixed precision (e.g. embeddings and output layer at 4 bits) if the size budget allows.
- **Data:** a few hundred hours sampled from ledger-approved sources Shruti also used, with eval clips removed by text and audio hash.
- **Also measure:** decoder depths 4 and 6 at 2 bits, which give a smaller tier for low-end phones.
- **Exit:** a 2-bit recipe and its measured cost: the WER of 2-bit versus 4-bit Shruti on every test set, with and without distillation, plus file size and phone RAM. This decides how akshar-lite is built. It never blocks the 4-bit akshar.
- **If 2 bits costs a lot even with distillation:** try mixed 2/4-bit and shallower decoders, and pick the best size/accuracy point for akshar-lite.

## Phase 3: Track B, akshar from Whistle

**Status (2026-11-07):** akshar v1 trained on dataset transcripts (D27) trails Shruti by 0.50 points of mean WER on the full test sets and beats it on FLEURS and MUCS. akshar v0 (1.3 behind) and akshar-lite v0 came first (D25). D26 cleared the last held sources, so accuracy is what stands between akshar and release. Details in [akshar.md](akshar.md).

- **Reproduce Shruti's recipe:**
  - Devanagari pieces in word-start and word-internal forms, with rows initialised from the mean of the byte pieces they replace.
  - A `hi` tag initialised from the mean of the existing tags.
  - Full fine-tune, then a second stage with more Indian English.
- **Convert every training transcript to the `mixed` style (D11):**
  - Put English words in Latin using dataset tags (IndicVoices and Rural Women brackets, Vaani braces).
  - Use a loanword lexicon for sets without tags.
  - Leave ambiguous words alone (बस can be "bus" or "just").
- **Build a clean data mix** (`admit` sources only, see [`../data/LICENSES.md`](../data/LICENSES.md)):
  - Transcribed Hindi: about 2,000 h before counting IndicVoices and Rural Women (Vaani, SPRING-INX, Kathbath, FLEURS).
  - About 14,900 h of untranscribed Vaani audio for pseudo-labels.
  - Plenty of general English.
  - Mine code-mixed utterances from IndicVoices, Vaani and SPRING-INX for Hinglish.
  - The gap to close is Indian-accented English. Options: resolve the NPTEL and Common Voice holds, pseudo-label English stretches inside the Indian corpora, or record our own with consent.
  - Pseudo-labels from permissively licensed teachers:
    - Hindi: Koyal-hi-120M (CC-BY-4.0, needs attribution; its punctuation and digits must be converted to our convention) and IndicConformer-600M (MIT).
    - English: Whisper large-v3 (MIT).
    - Shruti itself (Apache-2.0).
    - Record which teacher labelled every clip. Some teachers were trained on sources we hold (see "Teacher models" in [`../data/LICENSES.md`](../data/LICENSES.md)), so we must be able to drop one teacher's labels and retrain.
  - Silence and noise clips with empty targets.
  - Noise, reverb and speed augmentation.
- **Finish:**
  - Train akshar with 4-bit QAT throughout, as Shruti was.
  - Make akshar-lite with Track A's 2-bit recipe, using akshar's full-precision weights as the distillation teacher.
- **Exit (the release gate):**
  - **akshar:**
    - ≤ 30 MB file (hard cap 40 MB);
    - no test set worse than Shruti by more than 0.5 WER;
    - lower average WER than Shruti;
    - 0 invented words on silence.
  - **akshar-lite:**
    - about 20 MB, with the encoder at 4 bits (D24);
    - 0 invented words on silence;
    - WER published for every test set.
    - Provisional bar: average WER within 2 points of akshar. If it misses, ship akshar alone until it passes.

## Phase 4: Devices (Windows, Mac, phones)

Running the model needs no Linux (D12). Test the same `.cact` files on every target:

- **Windows x64**, and ARM64 if we can borrow a machine: use the `windows-x86_64` engine through both the C API and `needle --serve`. Shruti was never tested on Windows.
- **macOS on Apple silicon:** the `macos-arm64` engine. Intel Macs have no engine build (open question in D12).
- **Android:** a minimal harness on the C API (`needle_load`, `needle_transcribe`, `needle_stream_transcribe_process`), run on at least one low-end and one mid-range phone. Shruti's reference: Galaxy F62, 2 threads, 60 s in 9.5 s, 80 MB peak.
- **iOS:** the `ios-arm64` static library, once we have an iPhone to test on.
- **Long recordings:** apps must feed audio through the engine's streaming API in pieces. Loading a 1-hour recording at once costs about 115 MB on its own.
- **Exit:** on every target:
  - WER within 0.2 points of the Linux harness on the same clips;
  - akshar peaks at ≤ 90 MB RAM, and akshar-lite lower;
  - both are no slower than Shruti on the same device.

## Phase 5: Output styles (text layer)

This phase renders the model's `mixed` transcript as `roman`, `devanagari` or `mixed` (D11). It needs only text and a CPU, so it can run in parallel with training any time after Phase 0.

- **`devanagari` (English words → Devanagari):** a lexicon of standard Hindi spellings for common English words (laptop → लैपटॉप, office → ऑफिस). IndicXlit handles words not in the lexicon.
- **`roman` (Hindi words → Latin):**
  - Use the spellings people actually type (तुम्हारा → tumhara, है → hai), not scholarly ones with accents (tumhārā).
  - Take each word's most-typed spelling from a romanisation lexicon, and use IndicXlit for the rest.
  - Drop Hindi's silent "a" (कमरा → kamra, not kamara).
- **English words the model wrote in Devanagari** (लैपटॉप): look them up so that `roman` says "laptop", not a letter-by-letter spelling.
- **Options for every style:** capitals (`Laptop`), punctuation, numbers as digits.
- **Protected terms:** names, brands and code words pass through untouched. Apps can add their own terms (the same list used for keyword biasing).
- **Three-style reference set:** about 500 Hinglish test clips (Monsoon public, MUCS test, IndicVoices test), each reference written in all three styles and checked by a native speaker.
- **Budget (proposed):** the text layer adds at most about 5 MB to an app.
- **Exit (proposed):** each style's WER on the reference set is within 1 point of the script-agnostic WER, so rendering adds at most about one error per hundred words.

## Phase 6: Release

- Model card with harness numbers.
- NOTICE crediting Whistle, every dataset, every teacher model whose outputs went into training, and every text resource shipped in the text layer. Some text resources, such as Dakshina, are share-alike.
- The licence ledger and the eval scripts, so anyone can reproduce the numbers.

## Evaluation protocol

Fixed now so that akshar and Shruti are always scored the same way.

- **Normaliser:** Unicode NFC, lowercase, punctuation removed. Same function for references and hypotheses.
- **Scorer:** word-level edit distance (WER). We also report CER for Hindi sets. Monsoon uses the leaderboard's OIWER lattice scorer (`voi-oiwer`, MIT), which accepts listed spelling variants such as `इम्प्रूव | improve`. Report its corpus-level total, not a per-clip average.
- **Script-agnostic WER:** for Hinglish sets, also score after converting both sides to Roman, as Shruti did for MUCS. This measures what was heard separately from how it was spelled.
- **Per-style WER:** score each output style against references written in that style (the three-style reference set from Phase 5).
- **Decoding:** Needle engine, 5 beams, language flag set (`hi` for Hindi and Hinglish sets, `en` for Svarah and LibriSpeech), plus one auto-detect run.
- **Test sets (Shruti's clip counts):** Monsoon 753, FLEURS Hindi 418, Lahaja 6,152, MUCS 3,136, Svarah 6,656, LibriSpeech test-clean 2,620, 100 silent clips.
- **Contamination:** every test clip is blocked from training by normalised-text hash and audio hash, applied to all data, including data added later.
- **Reporting:** a WER is never quoted without its test set, normaliser and decoding settings.

## Compute plan (free only)

| Resource | When | Use |
|---|---|---|
| Laptop RTX 3050 4 GB (Ubuntu) | now | harness, data prep, Phase 1 (toolchain), Phase 2 (2-bit recipe), Phase 5 (text layer) |
| RTX 5080, 16 GB, on the owner's server (Ubuntu, SSH over Tailscale) | evenings, once its llama.cpp service is stopped (D20) | Phase 2 and Phase 3 training |
| Kaggle and Colab free tiers | now | overflow (JAX also runs on their TPUs) |
| Google Drive, 5 TB (the owner's plan) | now | pseudo-labels, packed training shards, checkpoints, backups. Training never reads from it directly. |
| Devices from the owner | by Phase 4 | low-end and mid-range Android phones, a Windows PC, an Apple-silicon Mac |

A 55M-parameter model is small; storage and audio loading will limit us before GPU maths does (D14).
- 2,732 h of 16 kHz 16-bit audio is about 315 GB as raw PCM and roughly half that as FLAC.
- The internet link is 100 Mbps, so moving 1 TB takes about a day. Each machine downloads source datasets itself, from Hugging Face at pinned revisions, instead of copying them between machines.
- The GPU server has about 45 GB free, so Phase 3's few hundred GB of audio needs another home before Phase 3 starts (D20).

## Risks

| Risk | Mitigation |
|---|---|
| 2-bit loses accuracy that distillation can't recover | Only akshar-lite is affected; akshar ships at 4 bits regardless. Mixed 2/4-bit and shallower decoders are fallbacks. |
| Our clean Hindi pool is smaller than Shruti's (Shrutilipi's 1,620 h is on hold) | Pseudo-label the ~14,900 h of untranscribed Vaani audio, add SPRING-INX, and ask AIR's News Services Division for permission. |
| Needle's open tooling lacks full-model training or 2-bit export for Whistle | The code is Apache-2.0 and the engine already runs 2-bit files. Extend the code; Phase 1 finds out early. |
| Clean Indian-English and Hinglish data is scarce (NPTEL, Common Voice and MUCS are on hold) | Mine code-mixed clips from admitted Indian corpora, pseudo-label, try to resolve the holds, and consider recording our own with consent. |
| Licence surprises (held sources, teacher-model licences) | Ledger gate (D5); Track A is research-only (D9). Exclude or replace anything unresolved before release. |
| Engine untested on our targets (Shruti tested Android, Linux, Mac and WASM; not Windows or iOS) | Phase 4 tests on real devices. |
| Engine builds come only from Cactus: Intel Macs have none, and it's unverified whether the engine source is public | Check github.com/cactus-compute/needle for the engine source. If it isn't there, ask Cactus about Intel builds or support Apple silicon only. |
