# The 2-bit recipe (Track A)

Phase 2 of the [roadmap](roadmap.md): find out what 2 bits costs and how to win it back, by training 2-bit versions of Shruti. The weights are research-only (D9); the recipe is what carries over to akshar-lite. Decisions D15, D16 and D22 record the method. Code: [`akshar/data/`](../akshar/data/) builds and labels the data, [`akshar/train/distill.py`](../akshar/train/distill.py) trains, and [`scripts/server_train.sh`](../scripts/server_train.sh) is the exact schedule, run on the RTX 5080 server (D20). [`scripts/phase2_track_a.sh`](../scripts/phase2_track_a.sh) is the laptop's version.

**Status (2026-11-04):** in progress. The laptop's 4 GB card ran out of memory on 1 November (D18), so the four runs started on the RTX 5080 server at 23:52 on 4 November (`scripts/server_train.sh`, D20). A 50-update test there peaked at 3.6 GB of GPU memory and trained on about 1,670 s of audio per second. The laptop copies back and scores each model (`scripts/pull_and_eval.sh`).

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

Results go here when the runs finish.

## Not measured here

- **Phone RAM** needs the Phase 4 devices.
- **Speed:** 2-bit files decode about 2.6 times slower than 4-bit in the x86 engine ([toolchain.md](toolchain.md)), so speed is measured per device in Phase 4.
