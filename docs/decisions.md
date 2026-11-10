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

- **Status:** amended by D19 (Shruti's four held sources may be trained on; release waits for legal review).
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

- **Status:** amended by D19 (Track B may train on Shruti's four held sources; release waits for legal review). D21 and D26 cleared all four sources; see D26.
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

## D15 (2026-10-31): Train and export with akshar's own quantiser and exporter

- **Decision:**
  - Full-weight training and `.cact` export go through `akshar.model` and `akshar.train`, built on needle 3.2.8's code ([toolchain.md](toolchain.md)).
  - The quantiser's Hadamard rotations run at HIGHEST precision, so training sees the codes the exporter writes.
  - Each weight matrix gets its own CQ width from a recipe. Phase 2 starts from Cactus's mixed recipe: the token embedding and the `mhc_phi_*` matrices at 4 bits, everything else at 2 bits.
  - needle's LoRA smoke test is skipped.
- **Why:**
  - needle's writer refuses widths other than 4, although the format and engine accept 1 to 4, and Cactus's own `whistle.cact` is mixed 2/4-bit.
  - At JAX's default precision on our GPU (TF32), training and export disagree on about 680 codes per million at 4 bits. At HIGHEST, under 1 per million.
  - Rebuilding `shruti.cact` from its checkpoint matched to about 2 codes per million. That proves the export path more strongly than a LoRA run would, and LoRA has no place in our plan.
- **Consequences:**
  - `akshar.model` calls private functions of needle's exporter. Before bumping `cactus-needle`, `tests/test_quant_export.py` must pass on the new version.
  - In the x86 engine, 2-bit files decode about 2.6 times slower than 4-bit (FLEURS: 17 against 46 times real time). akshar-lite's speed must be measured on every Phase 4 device, not assumed from its size.
- **Revisit if:** a needle release supports per-matrix widths itself, or changes the functions we wrap.

## D16 (2026-10-31): Track A trains on Shruti's own transcripts, distilled from Shruti as deployed

- **Decision:**
  - Track A's training targets are Shruti's own transcripts of the training audio, decoded by the engine with 5 beams and each source's language flag (`akshar.data.label`). Each dataset's own transcripts are kept only for the blocklist check.
  - The teacher runs as deployed: 4-bit weights and 8-bit activations, not full precision as the roadmap planned.
  - Data: IndicVoices Hindi (40 h), Rural Women ASR v2 Hindi (40 h), Kathbath Hindi (20 h), FLEURS Hindi train (3 h) and LibriSpeech train-clean-100 (22 h). All are `admit` or `admit, caveat` in the ledger, and every clip is checked against the test-set blocklist.
  - Models are compared on dev subsets (the first 200 to 300 clips of each test set, `python -m akshar.eval dev`). Final numbers come from full test sets.
- **Why:**
  - Track A's job is a 2-bit copy of Shruti. Shruti's output already follows our `mixed` style, while the datasets' transcripts follow other conventions: IndicVoices writes English words in Devanagari, Rural Women's carry noise tags, and some use digits. Only Phase 3's converters fix those.
  - On 368 LibriSpeech clips, the 4-bit teacher reproduces the engine's transcripts on 99.5% of tokens under teacher forcing, against 99.1% in full precision, with lower cross-entropy (0.034 against 0.037).
  - A 2-bit file decodes at about 17 times real time on the laptop, so a full seven-set evaluation takes about two hours per model. The dev subsets take about twelve minutes.
- **Consequences:**
  - The roadmap's "without distillation" arm is quantisation-aware training on Shruti's transcripts (alpha 0); "with distillation" adds the token-level KL term (alpha 0.5).
  - Track A models inherit Shruti's mistakes by design. They measure what 2 bits costs relative to Shruti, nothing more, and stay research-only (D9).
  - Measuring phone RAM moves to Phase 4, when the devices arrive.
- **Amends:** the Method and Data lines of the roadmap's Phase 2.

## D17 (2026-11-01): A text match removes a training clip only from three words up

- **Decision:**
  - A training clip whose audio matches a test clip's (the 16 kHz 16-bit PCM hash) is always removed.
  - A clip is removed for its text only when its normalised transcript has three or more words and matches a test clip's normalised text.
  - This applies from Phase 3 on. Phase 2's data was built under the old rule and is not rebuilt.
- **Why:**
  - In Phase 2 the text rule removed 1,251 IndicVoices clips. In the shard checked, every match was a one- or two-word utterance such as हाँ, नौ or सात that also occurs as a Lahaja reference. These are common words, not leaked test clips.
  - Short utterances are data akshar needs, for voice commands and answers.
  - A copy of a test clip is still caught by its audio, and a re-recording of a test sentence by its text.
- **Consequences:**
  - CLAUDE.md's contamination rule states the three-word floor.
  - A re-encoded copy of a one- or two-word test clip would get through. The risk is small, and its effect on a score is at most a word or two.
- **Revisit if:** a test set with many short references, such as a voice-command benchmark, joins the protocol.

## D18 (2026-11-01): Phase 2's training runs move to the 32 GB machine

- **Decision:** Phase 2's training runs (a1 to a4 in `scripts/phase2_track_a.sh`) restart on the 32 GB VRAM machine once its SSH access arrives. The laptop keeps the data, the teacher labels and the evaluation.
- **Why:**
  - On the laptop's 4 GB GPU, both full runs ran out of memory after 100 to 150 updates, when a 631 MB allocation failed. The full data mix has longer transcripts than the smoke tests did, so each batch shape needs more memory, and the card also drives the desktop.
  - Smaller batches would probably fit, but the 32 GB machine runs the same schedule faster with room to spare.
- **Consequences:**
  - Amends D14, which gave Phase 2 to the laptop.
  - Per D14, the GPU machine downloads the sources itself (`akshar.data.build` with the same hours and fractions), and the teacher labels travel as small files.
  - Nothing trained on the laptop is lost: the runs failed before writing a model, and Phase 2's data and labels are complete.
- **Revisit if:** the 32 GB machine is not available; then run Phase 2 on the laptop with about 60 s of audio per batch.

## D19 (2026-11-04): Shruti's four held sources join training; release waits for legal review

- **Status:** amended by D21 (Shrutilipi and AI4Bharat NPTEL admitted) and closed by D26 (MUCS 2021 and Common Voice admitted after legal guidance).
- **Decision:**
  - Phase 3 may train on the four sources Shruti used that the ledger holds: Shrutilipi, AI4Bharat NPTEL, MUCS 2021 Hindi-English and Common Voice's Indian-accented English. Their ledger status becomes `hold, train (D19)`.
  - No model trained on them is released until a lawyer has reviewed them. The project owner will arrange that review.
  - Every training clip keeps its source, so any one source can be dropped and the model retrained without it.
- **Why:**
  - They fill akshar's weakest spots: Indian-accented English (NPTEL, Common Voice), natural Hinglish (MUCS) and Hindi volume (Shrutilipi, 1,620 h). Without them akshar could trail Shruti on Svarah and MUCS.
- **Consequences:**
  - Amends D5 and D9. akshar is no longer commercially clean by construction; it becomes releasable only after the review.
  - If the review rejects a source, every model trained on it must be retrained without it.
  - Recommended, still open: also train a clean model without these sources. It is the fallback if a source is rejected, and it measures what the held sources are worth.
  - Common Voice downloads need the owner's Mozilla Data Collective account and acceptance of its terms.
  - They add roughly 150 to 200 GB of audio to Phase 3's storage needs.
  - Phase 2 (Track A) is unchanged.
- **Revisit after:** the legal review.

## D20 (2026-11-04): The GPU machine is an RTX 5080 with 16 GB, used in the evenings

- **Decision:** training runs on the RTX 5080 (16 GB) in the owner's server `codeacius-servo`, after 9 PM once its llama.cpp service is stopped. Work lives in `~/user1` there, and training starts only when no other process uses the GPU (`scripts/server_train.sh` checks).
- **Why:** the machine D14 described as a free 32 GB GPU has a 16 GB GPU that a llama.cpp service uses during the day. 16 GB is ample for Phase 2, which peaked at 2.5 GB.
- **Consequences:**
  - Amends D14 and D18.
  - The laptop reaches the server only through a Tailscale relay, at about 0.4 MB/s. Big files are fetched by the server itself from Hugging Face, at about 14 MB/s (`scripts/server_fetch.sh`).
  - The server's disk has about 45 GB free. Phase 3's audio, a few hundred GB with D19's sources, needs another home; still open.
- **Revisit if:** the server's GPU, disk or evening window changes.

## D21 (2026-11-04): CC-BY-4.0 datasets are admitted on the project owner's confirmation

- **Decision:**
  - The project owner has confirmed that the ledger's CC-BY-4.0 datasets can be used and distributed commercially.
  - Every training source labelled CC-BY-4.0 is admitted. That moves Shrutilipi and AI4Bharat NPTEL out of hold, and removes the caveats from Kathbath, Rural Women ASR v2, FLEURS and People's Speech's CC-BY configs.
  - Not covered, because they aren't CC-BY-4.0:
    - MUCS 2021, which is CC-BY-SA-4.0 at its source, and Common Voice, CC0 under Mozilla Data Collective terms, stay `hold, train (D19)`.
    - MUCS Hindi (OpenSLR 103) and Bhashini stay `hold`.
    - Non-commercial sources stay excluded.
  - Test sets stay test-only whatever their licence.
- **Why:** the owner's confirmation. The conflicts the ledger found stay in its notes for the record: All India Radio's terms for Shrutilipi's audio, and NPTEL's own licence badge.
- **Consequences:**
  - Amends D5 and D19.
  - A model trained without MUCS 2021 and Common Voice needs no further licence review under this ledger.
  - Track A stays research-only while MUCS 2021 and Common Voice are unresolved (D9).

## D22 (2026-11-04): Phase 2 trains on silence and noise with empty targets

- **Decision:**
  - Phase 2's runs add a sixth training set, `nonspeech`: 600 synthetic clips, about 1.8 h, made by `akshar.data.nonspeech` from a fixed seed.
  - A quarter each are digital silence, white noise, pink noise and 50 Hz mains hum. Noise is at -70 to -40 dB RMS relative to full scale, and clips are 1 to 20 s long.
  - Their target is empty: the model writes nothing and stops. The trainer now keeps clips with empty targets instead of skipping them.
- **Why:**
  - Shruti trained with silence and synthetic-noise clips that have empty transcripts, and it writes nothing on the harness's 100 silent clips.
  - Our five speech sets have no such clips: all 57,564 training clips had text. Two-bit training on speech alone could drift into inventing words on silence.
  - Shruti rounded to 2 bits with no training still writes nothing on the silent clips, so these clips keep a behaviour rather than teach a new one.
- **Consequences:**
  - Speech clips keep Shruti's transcripts as targets (D16). Shruti also writes nothing on silence, so the empty targets agree with it.
  - The harness's silence set, digital silence and faint noise, now resembles the training data, as it did for Shruti.
  - Every silence test clip is 4 s long and no training clip is, so no training clip can share a test clip's audio hash. A check against the blocklist's 19,735 audio hashes found no overlap either.
  - The clips contain no third-party audio. The ledger lists them as `admit`.
  - Phase 3 keeps them; its roadmap already plans silence and noise clips with empty targets.
- **Revisit if:** the dev silence test shows invented words, or a later noise test shows quiet speech being dropped.

## D23 (2026-11-05): GPU windows on the server end before 03:00

- **Decision:**
  - A training window that stops coot-moe ends, with coot-moe started again, before 03:00.
  - A window may cross 03:00 only once the owner has protected coot-moe from the nightly cleanup. `scripts/server_night.sh` refuses such a window unless `ALLOW_PRUNE_RISK=1` is set.
- **Why:**
  - The owner's crontab runs `~/scripts/cleanup_node.sh` at 03:00, which runs `docker system prune -af --volumes`. It deletes every stopped container, the images they used and unused networks.
  - On 5 November coot-moe was stopped at 01:00 for an overnight search. The 03:00 cleanup deleted it, its llama.cpp image and the `vllm_default` network, so it could not be started again at 04:58.
  - The image (by its recorded digest) and the network were restored. Recreating the container needs `~/Services/CootAI/deploy/run-coot.sh` with one of three env files, and only the owner knows which one is live.
- **Consequences:**
  - Amends D20, whose evening window had no end time.
  - A night of training on the server lasts about 9 PM to 3 AM, roughly five hours of GPU time after compile overheads, unless the owner protects coot-moe.
- **Revisit if:** the owner changes the cleanup to leave coot-moe alone, for example by skipping the prune while it is stopped or by filtering the prune on a label that coot-moe carries.

## D24 (2026-11-05): akshar-lite is about 20 MB, and Phase 2's mix adds NPTEL

- **Decision (the project owner's):**
  - akshar-lite keeps the encoder at 4 bits on top of Cactus's mix (recipe `mixed2-enc4`, 20.4 MB) instead of the 17.1 MB all-2-bit mix.
  - Phase 2's training mix adds about 30 h each of AI4Bharat NPTEL's English and Hindi lectures (`nptel_en`, `nptel_hi`), labelled by Shruti like the other sets.
- **Why:**
  - On the full test sets, the 20.4 MB model trails 4-bit Shruti by 1.4 points of mean WER, inside the roadmap's provisional 2-point bar. The 17.1 MB model trails by 2.3. The 20.4 MB model also decodes about twice as fast on x86.
  - Most of the remaining gap is on MUCS (Hinglish tech tutorials) and Svarah (Indian English), which the mix lacked. NPTEL's lectures cover Indian-accented English and Hindi full of English technical terms, and NPTEL is admitted (D21).
- **Consequences:**
  - Amends D10 (akshar-lite about 17 MB) and the data line of D16.
  - The release variants become about 20 MB (akshar-lite) and at most 30 MB (akshar).
  - NPTEL comes from the gated Hugging Face set `ai4bharat/NPTEL` at revision 9e488c7, so the server's account needs access.
- **Revisit if:** phones show the 4-bit encoder is not faster there, or a 17 MB model trained with NPTEL comes within 2 points of akshar.

## D25 (2026-11-06): akshar v0 is Whistle distilled from Shruti

- **Decision:**
  - The first akshar starts from Whistle's full-precision weights with Shruti's tokenizer (714 Devanagari pieces and `hi`), initialised as Shruti's report describes, and learns from Shruti: its transcripts as targets plus distillation from its token distributions.
  - It trains with 4-bit quantisation-aware training on Shruti-labelled audio from admitted sources and MUCS 2021. akshar-lite v0 is made from it with D24's recipe.
- **Why:**
  - The owner needed a working akshar by the morning of 6 November. Teacher labels need no transcript converters, and the Phase 2 trainer and data were ready.
  - The roadmap already lists Shruti as a teacher for pseudo-labels in Phase 3.
- **Consequences:**
  - akshar v0 trails Shruti by 1.3 points of mean WER on the full test sets and beats it on MUCS (akshar.md).
  - Like MUCS 2021, Shruti as a teacher falls under D19's legal review, because Shruti was trained on MUCS 2021 and Common Voice. akshar v0 is not released before that review.
  - Every clip keeps its source and its teacher, so either can be dropped and the model retrained from Whistle.
- **Revisit if:** the review rejects MUCS 2021 or Shruti as a teacher, or converted dataset transcripts are ready to train on.
- **Status:** amended by D26: the legal review cleared MUCS 2021 and Common Voice.

## D26 (2026-11-07): MUCS 2021 and Common Voice are cleared for commercial use

- **Decision:**
  - The project owner has received legal guidance that MUCS 2021 Hindi-English and Common Voice can be used commercially, and reported it on 7 November.
  - Both move from `hold, train (D19)` to `admit (D26)`. No training source in the ledger is `hold, train` any more.
  - Phase 3 trains on both, as Shruti did: MUCS 2021's training split (already in akshar v0) and Common Voice's Indian-accented English.
- **Why:** this is the legal review D19 asked for. The ledger keeps its notes on MUCS's share-alike licence and Mozilla Data Collective's terms as the record of what was checked.
- **Consequences:**
  - Amends D19, D21 and D25. akshar v0 and later versions are no longer held back by their data or by Shruti as their teacher. Shruti's weights are Apache-2.0, and both sources it was trained on that the ledger held are now admitted.
  - D9's stated reason, the four held sources inside Shruti's weights, no longer applies. Track A stays research-only until the owner decides otherwise.
  - Common Voice still has to be fetched. Its Hugging Face repository (`mozilla-foundation/common_voice_17_0`) accepts the server's account; Mozilla Data Collective's terms (no re-hosting, no re-identification) still apply to the files.
  - Test sets stay test-only whatever their licence, including MUCS 2021's test split.

## D27 (2026-11-07): akshar learns from datasets' own transcripts where they are verbatim

- **Decision:**
  - Training targets come from each dataset's own transcript, converted to the `mixed` style by `akshar.data.convert`, for sources whose transcripts are verbatim: IndicVoices, Kathbath, Rural Women ASR, Vaani, FLEURS and LibriSpeech.
  - MUCS 2021, NPTEL and Shrutilipi keep Shruti's transcripts as targets, because theirs are scripts or not verbatim. Silence keeps empty targets. Distillation from Shruti's token distributions stays at 0.5.
  - English words written in Devanagari become Latin only in spontaneous speech (IndicVoices, Vaani), through a lexicon learned from where Shruti writes them in Latin. Read speech keeps them in Devanagari, as Shruti writes them there and as the FLEURS and Lahaja references do.
- **Why:**
  - A student that learns only from Shruti's transcripts learns Shruti's mistakes. On IndicVoices and Rural Women the human transcripts are often more accurate than Shruti's.
  - Training akshar v1 on the converted transcripts for 2,400 updates lowered its mean WER on the full test sets from 22.45 to 22.28, most on the Hindi sets (akshar.md).
- **Consequences:**
  - Amends D25, whose targets were only Shruti's transcripts.
  - Sources with verbatim transcripts need no Shruti labelling, which removes the CPU labelling bottleneck for IndicVoices and Vaani. Shrutilipi still needs labels.
  - Every target records where it came from (`by` in labels-mix.jsonl), so either kind can be dropped.
- **Revisit if:** a longer run with a matching control shows no gain, or a test set regresses.

## D28 (2026-11-07): akshar v2 streams its data from Drive, and learns English from NPTEL's and LibriSpeech's transcripts

- **Decision:**
  - A run reads training shards from Google Drive while it trains. `akshar.train.feed` keeps about 10 GB of shards on the server's disk and swaps them as training goes; `akshar.train.distill --pool` takes 60% of its batches from them. Training still reads only local disk (D14).
  - The fixed data on disk is the Hinglish and English sets and silence: MUCS, NPTEL, LibriSpeech train-clean-100's labelled part, FLEURS and the silence clips. IndicVoices, Vaani, Shrutilipi's labelled part, Kathbath and Rural Women stream from Drive.
  - NPTEL's English references become targets, with numbers spelled out as words (`akshar.data.convert`); a reference that cannot be one falls back to Shruti's label, as for D27's sources. NPTEL's Hindi keeps Shruti's transcripts. In v2 the 53 h already on disk kept the Shruti labels they had; the new clips, which Shruti never labelled, used their references.
  - LibriSpeech's train-clean-360 and the rest of train-clean-100 join the stream with their own transcripts, which D27 already admits.
- **Why:**
  - akshar v1 trained on 524 h; Shruti trained on 2,732 h. The GPU had time for far more: v1's run processed about 6,000 h of audio, so it heard each clip about a dozen times. The limits were the server's disk, which holds about 530 h at a time, and, until D27, labelling by Shruti on the CPU.
  - Drive delivers audio several times faster than the GPU trains on it, so a small rotating buffer is enough.
  - English is where akshar is furthest behind (Svarah by 2.2, LibriSpeech by 1.0), with 75 h of English in v1's data. NPTEL's English part holds about 3,500 h of Indian-accented lectures. Its references are lecture subtitles, mostly verbatim, and Shruti's labels on the same clips have errors of their own ("x here" for "Excel").
- **Consequences:**
  - Amends D27 for NPTEL's English.
  - The server's copies of Kathbath, Rural Women and IndicVoices training audio were deleted, with the owner's approval; their copies are on Drive. Their validation clips stay on disk, so validation loss stays comparable with v1's.
  - Every shard a run streamed is logged with its source (`<pool>/fetched.jsonl`), and the run's record gives hours per source.
- **Revisit if:** v2 does worse than v1 on a set whose share of the data fell, or on English.

## D29 (2026-11-08): The converter is repaired from the label audit, and v4 trains on the repaired targets

- **Decision:**
  - A text-only audit of the training targets (about 1,000 lines over seven sources, read by Claude agents; inputs and results in `~/akshar-data/audit`) found that `akshar.data.convert` lost or mangled targets in ways a reader could see. The converter is repaired as follows, and the repaired targets are written beside the old ones as `labels-fixed.jsonl` (`--labels-name fixed`), so that a run reading `labels-mix.jsonl` is never disturbed.
  - Vaani: every square bracket is a sound and is removed; tags may hold spaces (`<static noise>`); a gloss may hold several words (`{blue colour}`), letters joined by `_` (`ए_सी {A_C}` is ac), a Devanagari correction (`शारी {सारी}`), and may follow an abbreviation written with dots (`ए.टी.एम {ATM}`); a sound written in the braces (`{breathing}`) is dropped; the annotators' misspellings in `GLOSS_TYPOS` are corrected; the lexicon's spelling replaces a gloss only when the two are spellings of one word (`{color}` becomes colour, `{said}` side; `{cap}` stays cap beside the lexicon's cab, `{lights}` stays plural).
  - Rural Women: its square brackets are glosses, apart from the few sounds in `RURAL_SOUNDS` (`[horn]`, `[clink]`); a Hindi word glossed in Latin keeps its Devanagari (`लाख [Lakh]`, `पनीर [paneer]`); misspelt glosses are corrected (`[Docter]`); a gloss of several words never spans a Hindi function word, so `और मोरक्कोनऑइल [moroccon oil]` replaces one token.
  - Lexicon lookup: a word is also found under a folded spelling (no nukta, short for long i, e for ai, o for au, anusvara for a half nasal, no reph), so `कन्फ़र्म`, `ड्राईवर`, `पर्सेन्ट` and `बौक्स` convert; English number words (`एटीन`, `थाउज़ेंड`, `पर्सेंट`) convert with any non-empty lexicon, so one number is never split across scripts; the Hindi slips `नो` beside a number word (for `नौ`) and `तेतीस` are repaired, in converted references and in Shruti's fallback labels.
  - English: Common Voice and LibriSpeech fold accented letters (`Castellón` is castellon) instead of losing the sentence. NPTEL drops the speaker label (`Jayanta Chatterjee:`); rejects a segment with a slash between short symbols (`1/r`), a slide-time tag, the sung motto, or a leading digit, and a segment that ends in a digit when the next segment of the video starts with one (a number split at its decimal point); reads Indian grouping (`1,05,000` is one lakh five thousand) and years after in, since, from and the like (`in 1995` is nineteen ninety five); leaves 1100-1999 elsewhere, 2010-2099 after such a word, malformed groupings (`1,0,5000`) and its mojibake (`PrimÕs`) to rejection.
  - akshar v3, which started training the same evening before these repairs were tested, keeps the old targets. v4 streams the `labels-fixed` targets.
- **Why:**
  - Vaani lost 106,114 of 424,964 references (25%) to markup the stripper could not parse; 9 of 10 sampled rejections were complete transcripts. The repaired converter accepts 402,038: 83,191 recovered, 3 lost, 15,952 changed (mostly `color` to colour, and misspelt glosses).
  - IndicVoices kept English words in Devanagari beside converted ones (`फ्री laptop स्कीम`); the folded lookup and the number words change 12,878 of 411,256 targets.
  - NPTEL's accepted targets held about 300 speaker labels, 292 maths slashes, 56 misread Indian groupings, 869 numbers that are years, and numbers split across segments. The repairs cost it 4% of its lines (nptel_en_b: 364,029 to 349,505, of which 5,626 are split numbers) and change 810.
  - Rural Women: 127 misspelt gloss tokens, `लाख` and `पनीर` in Latin, 17 brackets replacing the wrong word, `नो` for `नौ`. It now accepts 27,570 of 28,740 (33 recovered, 1 lost, 210 changed). Common Voice English recovers 77 sentences with accented names. Kathbath is unchanged.
- **Consequences:**
  - The eleven validation and fixed-data `labels-mix` files are untouched, so validation loss and the compile shapes stay comparable with v2 and v3.
  - `tests/test_convert.py` holds the audit's examples; `docs/research/` is unchanged (no new source).
  - Deferred, as style choices to test on dev first: the 750-word loanword list from the audit, `सर`/`बस`, foreign place names, ok/okay and etc/etcetera, plural glosses (`लाइटें {light}`), transcribers' matra slips, Kathbath's abbreviations (`डॉ`), and the 17 Rural Women brackets after the wrong word, which still replace it.
- **Revisit if:** v4 does worse than v3 on a dev set whose source was re-converted, or the recovered Vaani lines prove to carry wrong words.

## D30 (2026-11-09): akshar v3 trains on more Indian English, a fifth of it under the Hindi tag, and keeps a weight average

- **Decision:**
  - v3 starts from v2's weights and trains 110,000 more updates at a fresh learning rate of 2e-4 (cosine decay, no warm-up), with the same fixed data, validation clips and loss as v2 (4-bit QAT, datasets' transcripts where verbatim, distillation from Shruti at 0.5).
  - The stream from Drive adds Indian-accented English: Common Voice Scripted Speech 26.0, South Asian English (CC0, Mozilla Data Collective; admitted under D26), a second disjoint quarter of NPTEL's English lectures (`nptel_en_b`), and Common Voice 27.0 Hindi. The pool share rises to 70% of batches. The planned stream shares were Common Voice English 13%, NPTEL English 10% + 12%, LibriSpeech-360 5%, IndicVoices 22%, Vaani 17%, Shrutilipi 7%, Rural Women 6%, Kathbath 5%, Common Voice Hindi 3%, so that English would be about 38% of batches and Indian-accented English about 31%.
  - A fifth of each Indian English source's clips, chosen by a hash of the clip id, train under the `hi` tag with the same Latin target (`hi_tag_share` in `akshar.train.feed`), so that English spoken to the model in its default Hindi mode stays English.
  - From 60% of the run the trainer keeps an exponential moving average of the weights (decay 0.9998) and exports it beside the final weights (`--ema`), so both can be scored.
  - akshar-lite v3 is distilled from v3 with D24's recipe; a second candidate is distilled from the averaged weights.
- **Why:**
  - v2 beat Shruti on the mean but was 1.22 behind on Svarah, the only set more than 0.5 worse, and the owner found all models weak on their own Indian English. Shruti trained on Common Voice's Indian English, which v2 lacked.
  - The owner's own clips were decoded under the page's default `hi` flag, where Shruti's report measured 25.8 against 21.1 WER on Svarah; the `hi`-tagged share addresses that mode.
- **Consequences:**
  - v3 beats Shruti on every test set (Svarah 21.90 against 22.45), the first akshar to pass the release bar: mean 20.85 against Shruti's 21.78 and v2's 21.32. Under the `hi` flag Svarah is 28.47 (v2 32.86; Shruti's own figure 25.8).
  - The stream did not follow the planned shares. NPTEL's shards hold 26 to 32 hours each, a quarter of the 115-hour pool, and the feeder ranks sources by their absolute shortfall of hours in the pool, so the two NPTEL sources held about half the pool's hours throughout, Common Voice English a tenth, and the small Hindi sources starved: Kathbath was fetched twice, Common Voice Hindi once, Rural Women for 88 of 10,615 fetched hours. By fetched hours English was about 62% of the stream (about 53% of all batches), Hindi and Hinglish about 47%. Every Hindi set still improved.
  - Before v4 the feeder must either stage NPTEL in shards of a few hours, rank sources by relative shortfall, or let the trainer weight pool batches by source; and the run record should state the realised shares.
  - v3 saw about 3,670 distinct hours from the stream (IndicVoices 779, NPTEL English 771 + 820, Vaani 460, LibriSpeech-360 322, Shrutilipi 263, Common Voice English 158, Rural Women 78, Kathbath 14, Common Voice Hindi 5) plus the 230 fixed hours; about 10,600 hours passed through the pool, so streamed shards were seen about three times.
  - Common Voice enters NOTICE and the data tables; data/LICENSES.md already carries it as `admit (D26)`.
- **Revisit if:** v4 with the intended shares does worse on Svarah, or the `hi`-tagged English share hurts a Hindi set.

## D31 (2026-11-09): The run draws from the stream by source share, not by the hours a shard happens to hold

- **Decision:**
  - `akshar.train.feed` writes its sources' shares to `<pool>/shares.json`, and `akshar.train.distill` lists each source's pool clips as many times as its share divided by its hours in the pool demands (relative to the least-weighted source, rounded, at most 16 times), so the hours drawn per source follow the shares whatever the shard sizes.
  - The feeder picks the next shard from the source whose hours in the pool fall shortest of its share relative to that share, a source with nothing in the pool first, instead of by the absolute shortfall in hours.
- **Why:** D30 found that NPTEL's 26 to 32 hour shards took half of v3's pool and that the absolute ranking never reached Kathbath or Common Voice Hindi, so v3 trained on about 53% English against the 38% planned. Restaging NPTEL in small shards would mean re-uploading about 1,600 h to Drive; weighting the draws fixes the mix for any shard size.
- **Consequences:**
  - v4 trains with the planned shares. The run's log states the repeats in use at every refresh, and the feeder's fetched.jsonl still records what came down.
  - A source whose share is tiny for the hours it holds (a 30 h shard at 10%) is drawn from slowly, so its shards are seen partly before they leave the pool; the feeder's rotation brings them back.
  - Tested in the suite (`tests/test_feed.py`, `tests/test_loader.py`) and in a two-step CPU run on the laptop with a two-source scratch pool.
- **Revisit if:** v4's stream deviates from the planned shares by more than a few points (compare the log's repeats and fetched hours with the plan), or the repeats make the loader slow.

## D32 (2026-11-09): A curated list of English words, brands and place names that akshar writes in Latin

- **Decision:**
  - The converter's lexicon grows from the 917 words learned from Shruti to about 5,100 Devanagari spellings of 2,900 English words: technology, everyday objects, institutions and services, brands and spoken abbreviations, time and measure words, common Hinglish words, and place names in and outside India, written by eight Claude Opus 5.5 agents and reviewed against the data (`data/english-words/`, built by `scripts/build_english_words.py`).
  - A spelling converts only when it is safe: not on the agents' blocklist of Hindi look-alikes (पर, बस, सर, नो), not marked ambiguous, not a Hindi name that is a different word from the English one (चीन stays; दिल्ली becomes delhi), not an inflected Hindi form of a listed word (स्कूलों), and not a token that Vaani's and Rural Women's annotators leave in Devanagari at least 85% of the time. Place names convert to the name spoken (बनारस is banaras), never to a renamed official form.
  - Curated spellings are looked up exactly; only the learned words are also found under folded spellings (D29), and a blocked list protects Hindi words from both lookups. The lexicon file carries the counts that tell the two apart and the blocked list (`akshar.data.convert.load_lexicon_table`).
  - v4's Hindi targets (IndicVoices, Vaani, Kathbath, Rural Women, Common Voice Hindi) are re-converted with this lexicon, and the Shrutilipi targets, which are model transcripts, are rewritten token by token with it.
- **Why:**
  - The owner's own test of v3: Hindi is good, but English words inside Hindi speech and Indian-accented English are bad. The learned lexicon left most English words in Devanagari (the audit's lexicon gaps), so the model writes स्कूल beside school depending on the sentence, and the style the product promised (D11) was not what it learned.
  - Mining the lexicon from Shruti's output only reproduces Shruti's habits; a curated list says what the product should write.
  - The first merge showed why review against the data is needed: folded lookups turned ऐसी ("such") into ac 1,983 times and बेच ("sell") into batch, and the agents mapped बनारस to varanasi. Over 400,000 Hindi transcripts the final list converts 65,000 more tokens (about 1% of words, in 13% of lines), and the 700 most frequent conversions were read.
- **Consequences:**
  - FLEURS and Lahaja references keep loanwords and place names in Devanagari, so their WER may rise for v4; Monsoon's lattice scorer and MUCS accept Latin. The owner chose the product's behaviour over the table.
  - Fixed-data and validation `labels-mix` files are untouched; the re-converted targets are `labels-fixed2.*` beside them.
  - The list is a product asset as much as a training one: the text layer (Phase 5) can use it for the `devanagari` and `roman` styles.
- **Revisit if:** v4 writes Hindi words in Latin (a blocklist gap), or FLEURS and Lahaja fall by more than about a point.

## D33 (2026-11-10): akshar v4 is released as AksharSTT v4, with its final weights and Lite v4

- **Decision:**
  - The public name follows the internal version: akshar v4 is **AksharSTT v4** and akshar-lite v4 is **AksharSTT Lite v4**. The public line is v1 and v4; v0, v2 and v3 are called internal steps wherever they appear.
  - v4's final weights ship, not the moving average. Both score 20.78 on the six full sets; the final weights are better on Svarah (21.41 against 21.48), MUCS and LibriSpeech, which matter most to the owner, and Lite v4 was made from them.
  - Each variant has its own Hugging Face repo under byteknotlabs with the scored `.cact`, the record of its run, NOTICE, LICENSE and a card whose `model-index` carries the six scores; v4 also ships its full-precision weights. The cards are kept in `release/`.
  - The launch is one sequence: GitHub, then the Hugging Face repos go public, then the demo server serves v4, then the website goes live, then the GitHub release.
- **Why:**
  - The website and the demo API had already shown "v2" with v2's numbers, so calling v4 "v2" in public would give one name two sets of weights. One name per set of weights keeps the API ids, the docs and the records in step.
  - v4 meets the release bar (D13's harness; every set below Shruti on the same clips), and every training source is admitted in the ledger, with no `hold` row.
- **Consequences:** the website (akshar.byteknotlabs.com) and the demo API offer v4, Lite v4, v1 and Shruti. The Common Voice terms are met by a "machine-generated" disclosure in each card.
- **Revisit if:** a later version is released; it takes its internal number as its public name.

## Superseded from the earlier ChatGPT plan

- The project name BharatSTT is now **akshar**.
- "LoRA fine-tuning first" is superseded by D2. LoRA remains a plumbing smoke test only.
- "Peak memory below 256 MB" is now ≤ 90 MB for akshar and lower for akshar-lite (D10). Shruti measured 80 MB.
- The "825-clip FLEURS download" milestone is replaced by the D4 harness milestone.
