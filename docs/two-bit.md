# The 2-bit recipe (Track A)

Phase 2 of the [roadmap](roadmap.md): find out what 2 bits costs and how to win it back, by training 2-bit versions of Shruti. The weights are research-only (D9); the recipe is what carries over to akshar-lite. Decisions D15 and D16 record the method. Code: [`akshar/data/`](../akshar/data/) builds and labels the data, [`akshar/train/distill.py`](../akshar/train/distill.py) trains, and [`scripts/phase2_track_a.sh`](../scripts/phase2_track_a.sh) is the exact schedule.

**Status (2026-11-01):** in progress. The data is built and labelled, and the training runs in `scripts/phase2_track_a.sh` started on 1 November.

## Method

- **Student:** Shruti's full-precision weights, trained in full with quantisation-aware training at the target widths. The main recipe is Cactus's mixed one: token embedding and `mhc_phi_*` routing matrices at 4 bits, every other matrix at 2 bits (17.1 MB file).
- **Targets:** Shruti's own transcripts of the training audio, decoded by the engine as in evaluation (5 beams, the source's language flag). Datasets' own transcripts follow other conventions, and Track A's job is to copy Shruti, so Shruti's output is the target (D16).
- **Teacher:** Shruti as deployed, with 4-bit weights and 8-bit activations. On 368 LibriSpeech clips it reproduces the engine's transcripts on 99.5% of tokens under teacher forcing, against 99.1% in full precision.
- **Loss per token:** `(1 - alpha) × cross-entropy on Shruti's transcript + alpha × KL(teacher ‖ student)`. alpha 0 is training without distillation; alpha 0.5 adds it.
- **Batches:** clips grouped by length into four buckets (6, 12, 18 and 30 s), about 120 s of padded audio per batch and 4 batches per update. Learning rate 5e-5 with warm-up and cosine decay, 1,200 updates.

## Data

| Source | Ledger | Clips | Hours | Kind of speech |
|---|---|---|---|---|
| IndicVoices, Hindi | admit | 22,206 | 43.9 | read, extempore and conversational, from many districts |
| Rural Women ASR v2, Hindi | admit, caveat | 14,614 | 40.7 | rural extempore speech |
| Kathbath, Hindi | admit, caveat | 14,254 | 20.7 | read sentences recorded on phones |
| FLEURS, Hindi train | admit, caveat | 974 | 3.0 | read Wikipedia sentences |
| LibriSpeech train-clean-100 | admit | 6,116 | 22.0 | read US English |

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
