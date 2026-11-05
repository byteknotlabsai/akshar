# The 2-bit recipe (Track A)

Phase 2 of the [roadmap](roadmap.md): find out what 2 bits costs and how to win it back, by training 2-bit versions of Shruti. The weights are research-only (D9); the recipe is what carries over to akshar-lite. Decisions D15, D16 and D22 record the method. Code: [`akshar/data/`](../akshar/data/) builds and labels the data, [`akshar/train/distill.py`](../akshar/train/distill.py) trains, and [`scripts/server_train.sh`](../scripts/server_train.sh) is the exact schedule, run on the RTX 5080 server (D20). [`scripts/phase2_track_a.sh`](../scripts/phase2_track_a.sh) is the laptop's version.

**Status (2026-11-05):** the exit check is met: the 2-bit recipe and its cost on every test set, with and without distillation, are below. Phone RAM moved to Phase 4 (D16). Open: Hinglish and Indian-English data in the training mix, and whether akshar-lite may grow to about 20 MB. The first four runs trained on the RTX 5080 on the evening of 4 November; an overnight search followed on 5 November (`scripts/server_night.sh`).

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
- **Server scores** for the overnight runs of 5 November come from the server's CPU, with the same engine build. Its transcripts differ from the laptop's on about 14% of clips (a2: 245 of 1,800 dev clips), and set scores move by up to 0.4 points, though the means agree within 0.1. So server scores are compared only with server scores, including Shruti scored there (`s0-shruti-4bit`, `shruti-server`).

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

### Overnight search (5 November)

The owner lent the server's GPU until 7 AM. Two streams trained runs in parallel (`scripts/server_night.sh`), while the server's CPU scored each run on the dev subsets. Every run used distillation and D22's silence clips. Between runs, four things changed:
- **Learning rate:** 5e-5 (a2's), 2e-4 and 4e-4.
- **Length:** 1,200, 3,600 and 9,600 updates.
- **Distillation weight:** 0.5 and 0.9.
- **Which matrices get more bits:** the encoder, the decoder or the engram tables at 3 or 4 bits on top of Cactus's mix.

Dev subsets, scored on the server: normaliser `nfc-lower-punct2space-v1`, needle engine 3.3.3 with 5 beams, language `hi` for Monsoon, FLEURS, Lahaja and MUCS and `en` for Svarah and LibriSpeech. Monsoon is OIWER, and silence counts invented words.

| Run | What changed | File | Learning rate, updates | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | – | 29.5 MB | – | 22.30 | 20.23 | 22.83 | 51.32 | 21.29 | 5.74 | 0 | 23.95 |
| a2 | mixed 2-bit | 17.1 MB | 5e-5, 1,200 | 27.75 | 24.73 | 25.28 | 55.70 | 26.82 | 8.47 | 0 | 28.12 |
| b3 | distillation weight 0.9 | 17.1 MB | 5e-5, 1,200 | 27.70 | 23.98 | 25.32 | 55.39 | 26.48 | 8.32 | 0 | 27.86 |
| b1 | mixed 2-bit | 17.1 MB | 2e-4, 1,200 | 25.65 | 23.33 | 24.39 | 55.58 | 25.46 | 7.67 | 0 | 27.01 |
| b7 | mixed 2-bit | 17.1 MB | 4e-4, 1,200 | 25.58 | 23.51 | 24.25 | 55.32 | 26.17 | 7.16 | 0 | 27.00 |
| f1 | mixed 2-bit | 17.1 MB | 4e-4, 3,600 | 24.30 | 21.96 | 24.27 | 56.64 | 25.43 | 6.82 | 0 | 26.57 |
| c1 | mixed 2-bit | 17.1 MB | 4e-4, 9,600 | 23.62 | 21.76 | 23.84 | 57.37 | 25.74 | 6.94 | 0 | 26.55 |
| e2 | encoder at 3 bits | 18.7 MB | 2e-4, 1,200 | 24.71 | 22.72 | 24.31 | 53.51 | 23.30 | 6.84 | 0 | 25.90 |
| d2 | encoder at 4 bits | 20.4 MB | 2e-4, 1,200 | 24.04 | 22.31 | 23.82 | 53.06 | 23.89 | 6.54 | 0 | 25.61 |
| f4 | encoder at 4 bits | 20.4 MB | 4e-4, 3,600 | 23.56 | 22.29 | 23.58 | 54.05 | 23.95 | 6.31 | 0 | 25.62 |
| e3 | encoder and decoder at 3 bits | 20.8 MB | 2e-4, 1,200 | 24.07 | 21.82 | 23.80 | 52.94 | 22.81 | 6.20 | 0 | 25.27 |
| d1 | decoder at 4 bits | 21.2 MB | 2e-4, 1,200 | 25.80 | 21.78 | 24.33 | 52.85 | 23.55 | 7.19 | 0 | 25.92 |
| d3 | engram tables at 4 bits | 21.8 MB | 2e-4, 1,200 | 25.92 | 23.04 | 24.45 | 55.58 | 25.90 | 7.44 | 0 | 27.05 |

Two more runs trained but were not scored, because their 3-bit matrices decode too slowly: c2 (encoder at 3 bits, 9,600 updates) and f2 (encoder and decoder at 3 bits, 3,600 updates).

Decode speed on the server's CPU over the dev subsets. Other jobs ran at the same time, so these are rough:

| Recipe | File | Speed |
|---|---|---|
| Shruti, 4-bit | 29.5 MB | 66x real time |
| mixed 2-bit | 17.1 MB | about 18x |
| encoder at 4 bits | 20.4 MB | 37x |
| decoder at 4 bits | 21.2 MB | 19x |
| encoder at 3 bits | 18.7 MB | 11x |
| encoder and decoder at 3 bits | 20.8 MB | 7x |

What the search shows:
- **A higher learning rate is the cheapest win.** 2e-4 or 4e-4 instead of 5e-5 takes about 1.1 points off the mean at the same length.
- **Training longer helps the Hindi sets, not Hinglish or Indian English.** From 1,200 to 9,600 updates, Monsoon improves by 2 points on the dev subsets while the first 300 MUCS clips get 2 points worse. On the full MUCS set, c1 is level with a2 (below). The training mix has no Hinglish and no Indian-accented English, so training does not close those gaps.
- **A distillation weight of 0.9 changes little,** 0.26 points against 0.5.
- **The encoder suffers most at 2 bits,** then the decoder. The engram tables lose nothing.
- **3-bit matrices decode slowly in the x86 engine, and 4-bit ones fast.** The encoder at 4 bits makes the model both closer to Shruti and twice as fast as the 2-bit mix. 3 bits is out unless a device shows otherwise.
- **No model writes anything on the silent clips.**

### Full test sets

All seven sets, scored as in [eval.md](eval.md): normaliser `nfc-lower-punct2space-v1`, needle engine 3.3.3 with 5 beams, language `hi` for Monsoon, FLEURS, Lahaja and MUCS and `en` for Svarah and LibriSpeech. Monsoon is OIWER, and silence counts invented words. Each table compares runs scored on the same machine.

On the laptop:

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 20.63 | 20.36 | 26.31 | 35.17 | 22.47 | 5.95 | 0 | 21.81 |
| a1: 2-bit training, no distillation | 17.1 MB | 27.60 | 24.69 | 29.50 | 40.87 | 27.99 | 9.12 | 0 | 26.63 |
| a2: with distillation | 17.1 MB | 26.57 | 24.17 | 29.04 | 40.34 | 27.45 | 8.80 | 0 | 26.06 |
| c1: with distillation, learning rate 4e-4, 9,600 updates | 17.1 MB | 22.22 | 21.71 | 27.12 | 40.15 | 26.47 | 7.11 | 0 | 24.13 |

On the server:

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| f4: encoder at 4 bits, learning rate 4e-4, 3,600 updates | 20.4 MB | 21.63 | 21.84 | 26.94 | 37.84 | 24.48 | 6.51 | 0 | 23.21 |

- Distillation helps on every set, by 0.57 points on the mean (a1 against a2).
- The best 17.1 MB model, c1, trails Shruti by 2.3 points on the mean, against 4.3 for a2. The gap is largest on MUCS (5.0) and Svarah (4.0), the Hinglish and Indian-English sets our training mix lacks. On the other four sets it is 0.8 to 1.6 points.
- With the encoder at 4 bits, the 20.4 MB model trails by 1.4 points and decodes faster than the 17.1 MB mix.
- Shruti scores 21.81 on the laptop and 21.78 on the server, so the two machines agree on full sets to within 0.15 points per set.
- No model writes anything on silence.

### What this means for akshar-lite

- **The 2-bit recipe:** Cactus's mix, distillation at weight 0.5 from Shruti as deployed, learning rate 4e-4, and a few thousand updates. At 17.1 MB it trails 4-bit Shruti by 2.3 points on the full sets, just outside the roadmap's provisional 2-point bar.
- **The gap left is mostly Hinglish and Indian English.** Adding such speech to the training mix, labelled by the teacher, is the next lever; longer schedules no longer help there.
- **The encoder at 4 bits (20.4 MB) is inside the bar,** at 1.4 points, and decodes faster. Whether akshar-lite may grow from about 17 to about 20 MB is the owner's call (D10).
- **3-bit matrices decode too slowly in the x86 engine to use,** and the engram tables can stay at 2 bits.

## Not measured here

- **Phone RAM** needs the Phase 4 devices.
- **Speed:** 2-bit files decode about 2.6 times slower than 4-bit in the x86 engine ([toolchain.md](toolchain.md)), so speed is measured per device in Phase 4.
