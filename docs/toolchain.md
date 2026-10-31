# Training and export toolchain

Phase 1 of the [roadmap](roadmap.md): prove that we can train every weight of Whistle, quantise it the way the engine does, and export `.cact` files the engine runs and the [harness](eval.md) scores. Code: [`akshar/model/`](../akshar/model/) and [`akshar/train/`](../akshar/train/). Decision D15 records the choices.

**Status (2026-10-31):** the exit check passes. We train all 55M weights, fake-quantise at 4 and 2 bits to within rounding of the codes the exporter writes, and export 4-bit and 2-bit `.cact` files that the engine runs and the harness scores.

## What needle's code does

Read from `cactus-needle` 3.2.8 (Apache-2.0), the version `pyproject.toml` pins.

- **Quantisation (CQ).** Each weight matrix is split into groups of 128 along its input axis.
  - Each group is rotated by a normalised Walsh-Hadamard matrix, and its L2 norm is stored in FP16.
  - Each value of the unit-length group is snapped to the nearest entry of a Lloyd-Max codebook for a Gaussian: 4, 8 or 16 levels for 2, 3 or 4 bits.
  - The engine rebuilds a group as `(codebook[index] * norm) @ H`. Storage is the width plus 16 bits per 128 weights: 4.125 or 2.125 bits per weight.
  - Training uses the same function as a straight-through estimator. Activations and the KV cache are fake-quantised to 8 bits.
- **LoRA.** Adapters attach only to attention projections (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `out_proj`, including cross-attention). They are merged into the weights before quantisation. New vocabulary rows train too, but the MLPs, the existing embedding rows, the engram tables and the audio stem never adapt, which is one reason Shruti trained all weights.
- **The `.cact` writer.** A 120-byte header, a tensor directory, then 64-byte aligned tensors. The header carries the 2-, 3- and 4-bit codebooks, and each tensor record carries its own width. The format and the engine accept widths 1 to 4, but needle's writer refuses anything but 4.
- **Cactus's own 2-bit model.** The published `whistle.cact` is mixed: the token embedding and the six mHC routing matrices (`mhc_phi_*`) at 4 bits, the other 145 matrices at 2 bits. That is 2.3 bits per weight and 16.9 MB.
- **needle's trainer.** It trains LoRA only, in float32. It loads every clip into memory and pads each to 30 s and 320 tokens, which wastes compute on short clips and cannot hold more than a few dozen hours.

## What we built

- `akshar.model.checkpoint` loads and saves full-precision checkpoints with the tokenizer pieces they were trained with. Shruti's checkpoint records its 714 Devanagari pieces and `"mode": "full"` in its metadata.
- `akshar.model.export` writes `.cact` files through needle's writer, choosing a width for each matrix from a recipe: `uniform(4)`, `uniform(2)`, or `cactus_mixed` (Cactus's recipe above).
- `akshar.model.quant` is needle's quantiser with one fix (below) and a width per matrix.
- `akshar.train.finetune` trains every weight with quantisation-aware training at the recipe's widths, then saves the checkpoint and exports it.
- `akshar.data.fleurs` builds a training manifest from FLEURS Hindi's train split, keeping only clips whose transcript is pure Devanagari and checking each against the test-set blocklist.

## What we measured

All WER numbers: FLEURS Hindi test (418 clips) unless a row says otherwise, normaliser `nfc-lower-punct2space-v1`, needle engine 3.3.3 with 5 beams and language `hi`.

### Export is exact

- Rebuilding `shruti.cact` from `shruti.safetensors` with needle's own build gives the same header, codebooks, tokenizer and FP16 tensors. 25 of 11.8 million CQ codes differ, about 2 per million, plus a few norms by one FP16 step. Udgam Labs built on a Mac and we on x86, so this is floating-point rounding.
- Those 25 codes change 58 of the 418 FLEURS transcripts, yet WER moves only from 20.36 to 20.33. **Changes this small are noise:** beam search flips on tiny weight differences.
- `akshar.model.export` with `uniform(4)` writes a file byte-identical to needle's build (`tests/test_quant_export.py`).

### Training quantises the way the engine does, after one fix

JAX runs float32 matrix products in TF32 on this GPU by default, and needle's quantiser rotates each group with one. Training then picks different codes than the exporter writes:

| Shruti's 55M CQ weights | Codes that differ from the exporter's | Norms that differ |
|---|---|---|
| 4 bits, JAX default precision | 679 per million | 68,117 |
| 4 bits, rotation at HIGHEST precision | 0.7 per million | 24 |
| 2 bits, JAX default precision | 154 per million | 68,117 |
| 2 bits, rotation at HIGHEST precision | 0.1 per million | 24 |

`akshar.model.quant` runs both rotations at HIGHEST precision, so training sees the weights the engine will run, to rounding.

### 2 bits without retraining

| Shruti's weights | File | FLEURS WER |
|---|---|---|
| 4-bit, as released | 29.5 MB | 20.36 |
| 2-bit, Cactus's mixed recipe | 17.1 MB | 34.20 |
| 2-bit everywhere | 15.8 MB | 64.02 |

The engine loads and runs both 2-bit files. Quantising the embedding to 2 bits is what breaks the all-2-bit file, which is why Cactus keeps it at 4. Phase 2's job is to close the remaining gap with quantisation-aware training and distillation.

### Full-weight training fits the laptop

Training all 55M weights at 4 bits, 4 clips per batch, on the RTX 3050 (4 GB):
- Peak GPU memory 2.4 GB, so the card has room.
- About 97 s of audio per second, despite padding every clip to 30 s.

Two proof runs, starting from Shruti's full-precision weights. Training data: 2.83 h of FLEURS Hindi train, pure-Devanagari clips only, checked against the blocklist. 4 clips per pass, 4 passes per update.

| Model | Training | File | FLEURS WER | Monsoon OIWER |
|---|---|---|---|---|
| Shruti, 4-bit as released | none | 29.5 MB | 20.36 | 20.63 |
| Shruti, 4-bit | 20 updates at learning rate 1e-5 | 29.5 MB | 20.01 | not run |
| Shruti, mixed 2-bit | none | 17.1 MB | 34.20 | 56.90 |
| Shruti, mixed 2-bit | 150 updates at learning rate 5e-5 | 17.1 MB | 24.92 | 35.36 |

- FLEURS is in-domain for this training data, so its rows flatter the trained models. Monsoon, conversational phone audio, is out of domain.
- 150 updates, under five minutes of training on the laptop, recover about two thirds of the 2-bit loss on FLEURS and 60% on Monsoon, without distillation. Phase 2 adds distillation from the full-precision model and a few hundred hours of data.
- **2-bit files decode about 2.6 times slower than 4-bit in the x86 engine:** 17 to 19 times real time against 46. Cactus's 2-bit kernels are probably tuned for ARM, so akshar-lite's speed has to be measured on each device in Phase 4.

## Not done

- **needle's LoRA smoke test** (roadmap Phase 1, second bullet) was skipped. It would only re-prove the export plumbing, which the byte-identical rebuild above proves more strongly, and LoRA plays no part in our plan.
- **Length bucketing and a streaming loader.** Needed before Phase 2's few hundred hours; the in-memory, 30-second-padded loader is the main waste in training today.
