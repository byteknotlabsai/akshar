# The 2-bit recipe (Track A)

Phase 2 of the [roadmap](roadmap.md): find out what 2 bits costs and how to win it back, by training 2-bit versions of Shruti. The weights are research-only (D9); the recipe is what carries over to akshar-lite. Decisions D15, D16 and D22 record the method. Code: [`akshar/data/`](../akshar/data/) builds and labels the data, [`akshar/train/distill.py`](../akshar/train/distill.py) trains, and [`scripts/server_train.sh`](../scripts/server_train.sh) is the exact schedule, run on the RTX 5080 server (D20). [`scripts/phase2_track_a.sh`](../scripts/phase2_track_a.sh) is the laptop's version.

**Status (2026-11-05):** trained and scored on the dev subsets; a1 and a2 are being scored on the full test sets. The laptop's 4 GB card ran out of memory on 1 November (D18), so the four runs trained on the RTX 5080 server from 23:52 on 4 November to 00:37 on 5 November (`scripts/server_train.sh`, D20). The laptop copied back and scored each model (`scripts/pull_and_eval.sh`).

## Method

- **Student:** Shruti's full-precision weights, trained in full with quantisation-aware training at the target widths. The main recipe is Cactus's mixed one: token embedding and `mhc_phi_*` routing matrices at 4 bits, every other matrix at 2 bits (17.1 MB file).
- **Targets:** Shruti's own transcripts of the training audio, decoded by the engine as in evaluation (5 beams, the source's language flag). Datasets' own transcripts follow other conventions, and Track A's job is to copy Shruti, so Shruti's output is the target (D16). Silence and noise clips have empty targets, so the model learns to write nothing on them (D22).
- **Teacher:** Shruti as deployed, with 4-bit weights and 8-bit activations. On 368 LibriSpeech clips it reproduces the engine's transcripts on 99.5% of tokens under teacher forcing, against 99.1% in full precision.
- **Loss per token:** `(1 - alpha) × cross-entropy on Shruti's transcript + alpha × KL(teacher ‖ student)`. alpha 0 is training without distillation; alpha 0.5 adds it.
- **Batches:** clips grouped by length into four buckets (6, 12, 18 and 30 s). On the server, about 240 s of padded audio per batch and 2 batches per update; the laptop's version used 120 s and 4. Learning rate 5e-5 with warm-up and cosine decay, 1,200 updates.

## Data

| Source | Ledger | Clips | Hours | Kind of speech |
|---|---|---|---|---|
| IndicVoices, Hindi | admit | 22,206 | 43.9 | read, extempore and conversational, from many districts |
| Rural Women ASR v2, Hindi | admit (D21) | 14,614 | 40.7 | rural extempore speech |
| Kathbath, Hindi | admit (D21) | 14,254 | 20.7 | read sentences recorded on phones |
| FLEURS, Hindi train | admit (D21) | 974 | 3.0 | read Wikipedia sentences |
| LibriSpeech train-clean-100 | admit | 6,116 | 22.0 | read US English |
| Synthetic non-speech (D22) | admit | 600 | 1.8 | digital silence, white and pink noise, mains hum; empty targets |

- IndicVoices and Kathbath keep a fixed half of each shard's clips, so the hours come from more speakers.
- Every clip was checked against the blocklist of all 19,735 test clips, by normalised text and by audio. **1,251 IndicVoices clips were removed** by text; no other source lost any.
- Those matches are short common words, not shared sentences. In the one shard checked, all 420 matches were one- or two-word utterances (हाँ, नौ, सात) that also occur as Lahaja references. The rule removes them anyway, at a cost of under an hour of audio.

## How models are compared

- **Dev subsets** for choosing between runs: the first 200 to 300 clips of each test set, plus all 100 silence clips (`python -m akshar.eval dev`). Shruti's scores on the same clips come from its full runs.
- **Full test sets** for the final numbers, scored as in [eval.md](eval.md).

## Results

### Dev subsets

The first 200 to 300 clips of each test set and all 100 silence clips (`python -m akshar.eval dev`). Normaliser `nfc-lower-punct2space-v1`; needle engine with 5 beams; language `hi` for Monsoon, FLEURS, Lahaja and MUCS, `en` for Svarah and LibriSpeech. Monsoon is OIWER, and silence counts invented words. Shruti's row comes from its full runs, scored on the same clips.

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 22.24 | 20.35 | 22.87 | 51.53 | 21.45 | 5.71 | 0 | 24.03 |
| a0: mixed 2-bit, no training | 17.1 MB | 57.54 | 35.54 | 34.55 | 66.31 | 35.91 | 12.67 | 0 | 40.42 |
| a1: 2-bit training, no distillation | 17.1 MB | 29.05 | 25.06 | 25.95 | 56.10 | 27.16 | 8.92 | 0 | 28.71 |
| a2: with distillation | 17.1 MB | 28.19 | 24.67 | 25.45 | 55.74 | 26.70 | 8.38 | 0 | 28.19 |
| a3: a2 with 6 decoder layers | 15.9 MB | 33.00 | 27.64 | 27.52 | 59.11 | 28.06 | 10.14 | 0 | 30.91 |
| a4: a2 with 4 decoder layers | 14.7 MB | 38.97 | 30.23 | 30.12 | 64.52 | 33.50 | 11.91 | 0 | 34.88 |

- Training wins back about three quarters of what rounding to 2 bits costs. The best mean WER falls from 40.42 to 28.19, against Shruti's 24.03.
- Distillation helps a little on every set, 0.52 points on the mean. The full test sets must confirm a gap this small.
- Shallower decoders cost more than they save: 6 layers add 2.7 points for 1.2 MB less, and 4 layers add 6.7 points for 2.4 MB less.
- No model writes anything on the silent clips.
- The best 2-bit model trails 4-bit Shruti by 4.2 points. The same gap between akshar and akshar-lite would miss the roadmap's provisional bar of 2 points.

### Training

All four ran on the RTX 5080: 1,200 updates of about 480 s of padded audio each, learning rate 5e-5, on 130.7 h including D22's silence and noise clips.

| Run | Training time | Peak GPU memory | Validation cross-entropy at the end |
|---|---|---|---|
| a1 | 8 min | 3.9 GB | 0.171 |
| a2 | 10 min | 3.9 GB | 0.174, KL 0.096 |
| a3 | 9 min | 3.8 GB | 0.232, KL 0.148 |
| a4 | 9 min | 3.6 GB | 0.314, KL 0.220 |

Validation cross-entropy is measured against Shruti's transcripts. Before training it was 0.549. Times exclude about 5 minutes of compiling per run.

### Full test sets

a1 and a2 are being scored on all seven test sets; the numbers go here when they finish.

## Not measured here

- **Phone RAM** needs the Phase 4 devices.
- **Speed:** 2-bit files decode about 2.6 times slower than 4-bit in the x86 engine ([toolchain.md](toolchain.md)), so speed is measured per device in Phase 4.
