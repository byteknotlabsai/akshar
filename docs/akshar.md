# akshar (Track B)

Phase 3 of the [roadmap](roadmap.md): akshar itself, trained from Whistle rather than from Shruti's weights (D9). This page records how each version was made and how it scored.

**Status (2026-11-09):** akshar v3 beats Shruti on every test set: 20.85 mean WER on the full test sets against Shruti's 21.78 (20.78 with its averaged weights), both scored by our harness on the same clips, with Svarah's Indian English at 21.90 against 22.45. It is the first akshar to pass the roadmap's release bar. v3 trained from v2 on about 3,900 h, with more Indian-accented English (D30). v2 (21.32) beat Shruti's mean but trailed on Svarah; akshar v0, v1 and r1 (released as AksharSTT v1) came before it; see "Release" below.

## akshar v0

### How it was made

- **Start:** Whistle's own full-precision weights (`whistle.safetensors` from `Cactus-Compute/whistle`, fetched by needle), given Shruti's tokenizer: its 714 Devanagari pieces and the `hi` tag. Each new embedding row starts as the mean of the byte-piece rows Whistle's tokenizer would have used, and `hi` as the mean of the existing tags, exactly as Shruti's report describes (`akshar.model.track_b`, which uses needle's own code).
- **Teacher:** Shruti as deployed, 4-bit weights and 8-bit activations. The loss is Phase 2's: cross-entropy on Shruti's transcripts plus distillation from Shruti's token distributions, weight 0.5 (`akshar.train.distill`). Because the targets are Shruti's output, akshar writes the `mixed` style from the start, with no transcript converters.
- **Precision:** 4-bit quantisation-aware training throughout (recipe `w4`), as Shruti was trained.
- **Learning rate:** 8e-4, with warm-up and cosine decay over 56,000 updates of about 480 s of padded audio each. Two candidates trained side by side for the first 5,000 updates: 8e-4 reached validation cross-entropy 0.33 against 0.46 for 3e-4.
- **Schedule:** two GPU windows, 00:10 to 02:48 and 03:26 to 06:08, so that coot-moe ran through the owner's 03:00 cleanup (D23). About 5,300 hours of audio were processed in about 6 GPU-hours. Peak GPU memory 4.7 GB.

### Data

Every clip was labelled by Shruti. The first window trained on Phase 2's mix (186.6 h); from 03:26 the mix grew to 363.8 h:

| Source | Ledger | Clips |
|---|---|---|
| IndicVoices, Hindi (Phase 2's half) | admit | 22,206 |
| Kathbath, Hindi (labelled part) | admit (D21) | 19,163 |
| Rural Women ASR v2, Hindi (all) | admit (D21) | 28,740 |
| FLEURS, Hindi train | admit (D21) | 971 |
| MUCS 2021 Hindi-English, train | hold, train (D19) | 49,378 |
| NPTEL, English lectures | admit (D21) | 22,316 |
| NPTEL, Hindi lectures (labelled part) | admit (D21) | 18,096 |
| LibriSpeech train-clean-100 | admit | 6,116 |
| Silence and noise (D22) | admit | 600 |

The rest of Kathbath and NPTEL's Hindi lectures, and 817 h of IndicVoices, are staged on Google Drive for the next version.

### Results

Full test sets, scored on the server (normaliser `nfc-lower-punct2space-v1`, needle engine 3.3.3 with 5 beams, language `hi` for Monsoon, FLEURS, Lahaja and MUCS and `en` for Svarah and LibriSpeech; Monsoon is OIWER, silence counts invented words):

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| akshar v0 | 29.5 MB | 23.53 | 20.92 | 27.70 | 34.75 | 24.81 | 6.98 | 0 | 23.11 |
| akshar-lite v0 | 20.4 MB | 24.79 | 21.78 | 28.18 | 35.06 | 26.45 | 7.91 | 0 | 24.03 |

- akshar v0 trails Shruti by 1.3 points on the mean. It already beats Shruti on MUCS, the Hinglish set, by 0.5.
- The largest gaps are Monsoon's phone conversations (3.0) and Svarah's Indian English (2.4).
- It writes nothing on the silent clips.
- It does not yet meet the release bar for akshar: no set more than 0.5 worse than Shruti, and a lower mean.

Dev subsets, also on the server, for comparison with the Phase 2 models:

| Model | File | Mean WER, dev subsets |
|---|---|---|
| Shruti, 4-bit | 29.5 MB | 23.95 |
| akshar v0 | 29.5 MB | 25.51 |
| akshar-lite v0 | 20.4 MB | 26.38 |
| f4, Track A, encoder at 4 bits | 20.4 MB | 25.62 |

## akshar-lite v0

D24's recipe made from akshar: Cactus's 2-bit mix with the encoder at 4 bits (20.4 MB), starting from akshar v0's full-precision weights and distilled from akshar v0 as deployed, learning rate 4e-4, 3,600 updates on the same 363.8 h. On the full test sets it trails akshar v0 by 0.9 points of mean WER, inside the roadmap's provisional 2-point bar, and it too edges Shruti on MUCS (35.06 against 35.26). It writes nothing on silence.

## akshar v1

### How it was made

- **Start:** akshar v0's full-precision weights, trained further with v0's loss and recipe: 4-bit quantisation-aware training, Shruti's transcripts as targets plus distillation from Shruti at 0.5.
- **Learning rate:** 4e-4, half of v0's peak, with warm-up over 3,200 updates and cosine decay over 64,000 updates (run `m1-akshar-v1`).
- **Data:** v0's 364 h, the 126 h of Kathbath and NPTEL Hindi that v0 never reached, and 39 h of new IndicVoices clips. The IndicVoices clips came from Drive in two slices and were labelled by Shruti overnight. That made 504.6 h in the first window and 524.3 h in the second, which added a second slice.
- **Schedule:** two GPU windows, 00:24 to 02:45 and 03:05 to 06:29, with coot-moe running through the owner's 03:00 cleanup (D23).
  - Each window spent 12 to 15 minutes compiling before its first update, against 7.5 the night before, because the owner's k3s server was using about five cores.
  - About 3.4 updates a second otherwise.
- **Validation cross-entropy against Shruti's labels:** 0.1396 at the start, 0.1496 after the warm-up, 0.1233 at the end.

### Results

Full test sets on the server, with the same harness settings as v0 above:

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| akshar v0 | 29.5 MB | 23.53 | 20.92 | 27.70 | 34.75 | 24.81 | 6.98 | 0 | 23.11 |
| akshar v1 | 29.5 MB | 22.07 | 19.83 | 26.89 | 34.46 | 24.42 | 7.01 | 0 | 22.45 |

- v1 beats Shruti on FLEURS (by 0.38) and MUCS (by 0.80).
- It improves on v0 everywhere except LibriSpeech, which is level. Monsoon's phone conversations improve most, by 1.46.
- The gaps left: Svarah's Indian English 1.97, LibriSpeech 1.06, Monsoon 1.53 and Lahaja 0.60.
- It writes nothing on the silent clips.

## v1 trained on dataset transcripts (r1)

- **What changed:** the targets. Where a dataset's own transcript is verbatim, it replaces Shruti's, converted to the `mixed` style by `akshar.data.convert` (D27). The sources are IndicVoices, Kathbath, Rural Women, FLEURS and LibriSpeech.
  - Noise tags are removed, and Rural Women's bracketed English spellings are kept.
  - English words that IndicVoices writes in Devanagari become Latin, through a 917-word lexicon. It was learned from the places where Shruti writes them in Latin.
  - Read speech (Kathbath, FLEURS) keeps its Devanagari loanwords, as Shruti and the FLEURS and Lahaja references write them.
  - MUCS, NPTEL and silence keep their old targets. Distillation from Shruti stays at 0.5.
- **Coverage:** converted transcripts were the targets for 96 to 100% of those sources' clips. The rest have unknown-word marks or broken markup and keep Shruti's.
- **Training:** 2,400 updates from v1 at learning rate 1e-4 (run `r1-akshar-v1-ref`, 06:29 to 06:59).

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| akshar v1 | 29.5 MB | 22.07 | 19.83 | 26.89 | 34.46 | 24.42 | 7.01 | 0 | 22.45 |
| r1, v1 trained on dataset transcripts | 29.5 MB | 21.78 | 19.37 | 26.49 | 34.41 | 24.68 | 6.92 | 0 | 22.28 |

- r1 beats v1 on five of the six sets. Its mean, 22.28, is 0.50 behind Shruti's.
- The Hindi sets gain most: Monsoon by 0.29, FLEURS by 0.46 and Lahaja by 0.40. Svarah, whose English data still learns from Shruti's transcripts, is 0.26 worse.
- Against Shruti, r1 is ahead on FLEURS (by 0.84) and MUCS (by 0.85), and Lahaja is within 0.20.
- r1 also trained 2,400 updates longer than v1, so not all of the 0.17 comes from the targets.
- The file is `models/akshar/akshar-v1-ref.cact`.

## akshar v2

### How it was made

- **Start:** r1 (AksharSTT v1), trained further with r1's loss: 4-bit quantisation-aware training, datasets' own transcripts as targets where they are verbatim (D27), and distillation from Shruti at 0.5.
- **Data:** about 3,000 h of distinct audio, against v1's 524 h, streamed from Google Drive while the run trained (D28).
  - `akshar.train.feed` kept 7 to 10 GB of Drive shards on the server's disk and swapped them as training went; `akshar.train.distill --pool` took 60% of its batches from them. Drive delivered about 800 h of audio an hour, more than the run took.
  - The other 40% came from 230 h kept on disk: MUCS, NPTEL's Hindi and English lectures (the parts Shruti labelled), LibriSpeech train-clean-100 (the part v1 used), FLEURS and the silence clips.
  - The validation clips are v1's: the same 1% of every source, so the loss below compares with v1's.
- **Targets:** datasets' own transcripts for IndicVoices, Vaani, Kathbath, Rural Women, LibriSpeech and the new NPTEL English clips, whose numbers are spelled out as words (D28). Shruti's transcripts for MUCS, NPTEL's labelled clips and Shrutilipi.
- **Learning rate:** 3e-4, with warm-up over 3,900 updates and cosine decay over 78,000 updates of about 480 s of padded audio each (run `v2-akshar`).
- **Schedule:** two GPU windows, 22:37 to 02:45 and 03:06 to 05:30, with coot-moe running through the owner's 03:00 cleanup (D23).
  - About 3.5 updates a second; 7,240 h of audio processed, 4,250 h of it from the stream.
  - The first start compiled for 11 minutes. The resume after 03:00 compiled from JAX's cache in 22 seconds.
- **Validation cross-entropy:** 0.2327 at the start (r1), 0.2457 after the warm-up, 0.2129 at the end.

Streamed from Drive, by source (each shard counted once; streamed shards were seen about 1.5 times on average):

| Source | Ledger | Hours | Targets |
|---|---|---|---|
| IndicVoices, Hindi | admit | 779 | own transcripts |
| NPTEL, English lectures (new clips) | admit (D21) | 771 | own transcripts, numbers spelled out |
| Vaani, transcribed Hindi | admit | 460 | own transcripts |
| LibriSpeech train-clean-360 | admit | 292 | own transcripts |
| Shrutilipi, Hindi (labelled part) | admit (D21) | 263 | Shruti's |
| Kathbath, Hindi | admit (D21) | 130 | own transcripts |
| Rural Women ASR v2, Hindi | admit, caveat | 79 | own transcripts |

- **Shares in the stream:** IndicVoices 34%, Vaani 26%, NPTEL English 12%, Shrutilipi 12%, Kathbath 8%, Rural Women 8%, LibriSpeech 8%.
- **One change at 02:50:** the dev check at update 48,000 showed Svarah flat since update 29,000 while FLEURS and Lahaja led Shruti. For the second window NPTEL English went to 18% and LibriSpeech-360 to 8%, taken from Shrutilipi (10%) and Kathbath (6%).

### Results

Full test sets on the server, with the same harness settings as v0 above:

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| r1 (AksharSTT v1) | 29.5 MB | 21.78 | 19.37 | 26.49 | 34.41 | 24.68 | 6.92 | 0 | 22.28 |
| **akshar v2** | 29.5 MB | **20.45** | **18.42** | **25.05** | **34.30** | 23.67 | 6.03 | 0 | **21.32** |

- v2's mean is 0.46 below Shruti's. It is the first akshar to beat Shruti on the mean.
- It is ahead of Shruti on four sets: FLEURS by 1.79, Lahaja by 1.24, MUCS by 0.96 and Monsoon by 0.09. LibriSpeech is level (0.08 behind).
- Svarah's Indian English is still 1.22 behind, though 1.01 better than r1. Shruti trained on Common Voice's Indian English, which akshar does not have yet.
- v2 improves on r1 on every set, by 0.96 points of mean, and writes nothing on the silent clips.
- Peak memory while decoding is the same as Shruti's (122 MB on Svarah).
- The file is `models/akshar/akshar-v2.cact`.

Dev subsets during the run (server, mean of the six speech sets):

| Model | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Mean |
|---|---|---|---|---|---|---|---|
| Shruti, 4-bit | 22.30 | 20.23 | 22.83 | 51.32 | 21.29 | 5.74 | 23.95 |
| r1 | 23.30 | 19.40 | 23.68 | 51.51 | 24.38 | 6.76 | 24.84 |
| v2 at update 5,000 (peak learning rate) | 24.16 | 20.70 | 23.90 | 50.49 | 23.92 | 6.77 | 24.99 |
| v2 at update 29,000 | 22.99 | 19.50 | 22.34 | 50.75 | 23.24 | 6.02 | 24.14 |
| v2 at update 48,000 (end of window 1) | 22.53 | 18.99 | 22.32 | 50.75 | 23.21 | 6.00 | 23.97 |

On the dev subsets r1 looked 0.89 behind Shruti; on the full sets it was 0.50 behind. The dev subsets are small (200 to 300 clips) and only show direction.

## akshar v3

### How it was made

- **Start:** v2's weights, trained 110,000 more updates with v2's loss (4-bit quantisation-aware training, datasets' own transcripts where verbatim, distillation from Shruti at 0.5) and the same fixed data and validation clips, so its validation loss compares with v2's (D30).
- **Learning rate:** 2e-4 with cosine decay and no warm-up (a warm restart: validation cross-entropy went from v2's 0.2129 to 0.2253 by update 8,000 and came back to 0.2114 at the end). From update 66,000 an exponential moving average of the weights (decay 0.9998) was kept and exported beside the final weights; its validation cross-entropy is 0.2112.
- **Data:** the stream from Drive now adds Indian-accented English: Common Voice Scripted Speech 26.0, South Asian English (160 h), a second quarter of NPTEL's English lectures (`nptel_en_b`, 867 h) and Common Voice 27.0 Hindi (10 h), all built on 8 November. The pool took 70% of batches (v2: 60%). A fifth of each Indian English source's clips trained under the `hi` tag with the same Latin target, so that English spoken to the model in its default Hindi mode stays English.
- **Schedule:** two GPU windows, 19:28 to 02:45 and 03:05 to 04:42, with coot-moe running through the owner's 03:00 cleanup (D23); about 3.45 updates a second, 110,000 updates of about 480 s of padded audio. The resume compiled from JAX's cache in 40 s. Runs `v3-akshar`, then `l4-akshar-lite-v3` and `l4-akshar-lite-v3-ema` (18 min each) in the same window.

Streamed from Drive, by source (each shard counted once):

| Source | Ledger | Hours | Targets |
|---|---|---|---|
| NPTEL, English lectures (two quarters) | admit (D21) | 771 + 820 | own transcripts, numbers spelled out |
| IndicVoices, Hindi | admit | 779 | own transcripts |
| Vaani, transcribed Hindi | admit | 460 | own transcripts |
| LibriSpeech train-clean-360 | admit | 322 | own transcripts |
| Shrutilipi, Hindi (labelled part) | admit (D21) | 263 | Shruti's |
| Common Voice 26.0, South Asian English | admit (D26) | 158 | own transcripts |
| Rural Women ASR v2, Hindi | admit, caveat | 78 | own transcripts |
| Kathbath, Hindi | admit (D21) | 14 | own transcripts |
| Common Voice 27.0, Hindi | admit (D26) | 5 | own transcripts |

- **Planned shares in the stream:** Common Voice English 13%, NPTEL English 10% + 12%, LibriSpeech-360 5%, IndicVoices 22%, Vaani 17%, Shrutilipi 7%, Rural Women 6%, Kathbath 5%, Common Voice Hindi 3%: about 38% English in all batches.
- **What the stream did:** NPTEL's shards hold 26 to 32 h each, a quarter of the 115 h pool, and the feeder ranks sources by their absolute shortfall of hours, so the two NPTEL sources held about half the pool's hours throughout and the small Hindi sources starved (Kathbath was fetched twice in the night, Common Voice Hindi once). By fetched hours the stream was NPTEL English 49%, IndicVoices 20%, Vaani 11%, Common Voice English 10%, Shrutilipi 6%, LibriSpeech 3%, Rural Women 1%: about 62% English, so about 53% of all batches. About 10,600 h passed through the pool over the night; each streamed shard was seen about three times. The feeder is to be fixed before v4 (D30).
- **Dev check at update 33,000** (dev subsets, server): Monsoon 22.52, FLEURS 18.66, Lahaja 22.03, MUCS 51.15, Svarah 22.00, LibriSpeech 6.10, mean 23.74; ahead of v2 at its update 29,000 (24.14) on every set but MUCS and LibriSpeech, so the shares were left alone.

### Results

Full test sets on the server, with the same harness settings as v0 above:

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| akshar v2 | 29.5 MB | 20.45 | 18.42 | 25.05 | 34.30 | 23.67 | 6.03 | 0 | 21.32 |
| akshar v3, final weights | 29.5 MB | 19.82 | 18.32 | 24.88 | 34.24 | 21.90 | 5.93 | 0 | 20.85 |
| **akshar v3, averaged weights** | 29.5 MB | **19.70** | **18.27** | **24.87** | **34.09** | **21.85** | **5.88** | 0 | **20.78** |

- v3 is the first akshar ahead of Shruti on every set: Monsoon by 0.72, FLEURS by 1.89, Lahaja by 1.41, MUCS by 1.02, Svarah by 0.55 and LibriSpeech by 0.02; its mean is 0.93 below Shruti's and 0.47 below v2's.
- Svarah gained 1.77 on v2; the Hindi sets gained 0.10 to 0.63 even though the stream gave them less than planned.
- Svarah decoded with the `hi` flag, outside the protocol: v2 32.86, v3 28.47 (Shruti's own report measured 25.8 against 21.1 with `en`). The `hi`-tagged English share narrowed the gap between the two modes from 9.2 to 6.6 points.
- The averaged weights (the moving average from update 66,000) beat the final weights on every set, by 0.07 of mean. They are the release candidate.
- Both write nothing on the silent clips. The files are `models/akshar/akshar-v3-ema.cact` (averaged) and `akshar-v3.cact` (final).

## akshar-lite v3

D24's recipe made from akshar v3 (run `l4-akshar-lite-v3`): Cactus's 2-bit mix with the encoder at 4 bits (20.4 MB), starting from v3's full-precision weights and distilled from v3 as deployed, learning rate 4e-4, 3,600 updates on v3's fixed data and stream (18 minutes, 426 h streamed). A second candidate, `l4-akshar-lite-v3-ema`, was made the same way from v3's averaged weights.

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| akshar v3, averaged weights | 29.5 MB | 19.70 | 18.27 | 24.87 | 34.09 | 21.85 | 5.88 | 0 | 20.78 |
| akshar-lite v2 | 20.4 MB | 22.45 | 19.73 | 25.06 | 34.63 | 25.07 | 6.74 | 0 | 22.28 |
| **akshar-lite v3** | 20.4 MB | 22.07 | 19.27 | 24.53 | 34.79 | 23.47 | 6.53 | 0 | **21.78** |
| akshar-lite v3, from the averaged weights | 20.4 MB | 21.27 | 19.57 | 24.97 | 34.55 | 23.05 | 6.59 | 0 | 21.67 |

- akshar-lite v3's mean equals Shruti's (21.78) in a file 9 MB smaller; it improves on akshar-lite v2 by 0.50 and trails akshar v3 by 0.93, inside the roadmap's 2-point bar.
- It beats Shruti on FLEURS (by 0.94), Lahaja (by 1.76) and MUCS (by 0.47); Monsoon is 1.53 behind, Svarah 1.02 and LibriSpeech 0.58.
- The candidate from the averaged weights has the lower mean (21.67): better on Monsoon (0.80), Svarah (0.42) and MUCS, worse on FLEURS, Lahaja and LibriSpeech. Which one ships is a choice between Hindi conversation and read Hindi.
- Both write nothing on silence. The files are `models/akshar/akshar-lite-v3.cact` and `akshar-lite-v3-ema.cact`.

## akshar-lite v2

D24's recipe made from akshar v2: Cactus's 2-bit mix with the encoder at 4 bits (20.4 MB), starting from v2's full-precision weights and distilled from v2 as deployed, learning rate 4e-4, 3,600 updates on v2's fixed data and stream (run `l3-akshar-lite-v2`, 05:30 to 05:51, 411 h streamed).

| Model | File | Monsoon | FLEURS Hindi | Lahaja | MUCS | Svarah | LibriSpeech clean | Silence | Mean WER |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit as released | 29.5 MB | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| akshar v2 | 29.5 MB | 20.45 | 18.42 | 25.05 | 34.30 | 23.67 | 6.03 | 0 | 21.32 |
| akshar-lite v0 | 20.4 MB | 24.79 | 21.78 | 28.18 | 35.06 | 26.45 | 7.91 | 0 | 24.03 |
| **akshar-lite v2** | 20.4 MB | 22.45 | 19.73 | 25.06 | 34.63 | 25.07 | 6.74 | 0 | **22.28** |

- akshar-lite v2 trails akshar v2 by 0.96 points of mean, inside the roadmap's 2-point bar, and improves on akshar-lite v0 by 1.75.
- Its mean equals r1's (AksharSTT v1, 22.28) in a file 9 MB smaller.
- It beats Shruti on FLEURS (by 0.48), Lahaja (by 1.23) and MUCS (by 0.63), and writes nothing on silence.
- The file is `models/akshar/akshar-lite-v2.cact`.

## Running it

The files are `models/akshar/akshar-v3.cact`, `akshar-v3-ema.cact`, `akshar-lite-v3.cact`, `akshar-lite-v3-ema.cact`, `akshar-v2.cact`, `akshar-lite-v2.cact`, `akshar-v1.cact`, `akshar-v0.cact` and `akshar-lite-v0.cact` on the laptop (not committed), and on Drive under `akshar/phase3-runs/`. Any needle engine runs them like Shruti:

```python
from needle import Whistle
print(Whistle("models/akshar/akshar-v1.cact").transcribe(samples, language="hi")["text"])
```

To try them in a browser, start the local page and open http://127.0.0.1:8765:

```bash
uv run python -m akshar.ui
```

- **Transcribe:** record from the microphone or open an audio file, and every `.cact` under `models/` transcribes it, Shruti included. Each transcript shows its word timings, confidences and decoding speed. Paste what was said to get the word error rate with the harness's normaliser.
- **Test results:** the full test-set tables from `runs/server-full/` and `runs/`, and every clip behind them. Listen to a clip, see each model's errors word by word, and decode it again on this machine.
- **API:** how an app calls the API server below, with this machine's address.

The page, the engine and the results stay on this machine; the server listens on 127.0.0.1 only.

### On a server

`akshar.serve` is the same engine behind an HTTP API, with no page: an app posts a recording to `/v1/transcribe` and gets the transcript as JSON. `scripts/build_deploy.py` packs it into one folder to unzip and run on another machine:

```bash
uv run python scripts/build_deploy.py --zip   # ~/Projects/akshar-deploy and akshar-deploy.zip
```

The folder holds:
- the server;
- the `.cact` files;
- the engine library our scores were measured with (Linux x86-64);
- `settings.env`: the models offered, the defaults, API keys, allowed websites and limits;
- a `run.sh` that installs its packages on the first start.

Its README is [deploy/README.md](../deploy/README.md), with the models' stored scores filled in.

The package's NOTICE credits every training source, as a release of these models must.

## Release

D26 cleared the two sources that held akshar v0 back: MUCS 2021, in its training data, and Common Voice, inside its teacher Shruti. Both were cleared on the owner's legal guidance. akshar v0 and akshar-lite v0 are no longer held back by their data or by their teacher. Before their weights are published, NOTICE must credit every training dataset (data/LICENSES.md).

What stands between akshar and a release is now accuracy. The roadmap's bar is a lower mean than Shruti's with no set more than 0.5 worse. akshar v2 has the lower mean (21.32 against 21.78) and meets the bar on every set but Svarah, which is 1.22 behind. akshar-lite v2 meets its own bar, within 2 points of akshar.

On 7 November the owner released r1 as **AksharSTT v1**, before that bar is met: [huggingface.co/byteknotlabs/AksharSTT-V1](https://huggingface.co/byteknotlabs/AksharSTT-V1), Apache-2.0.
- **Contents:** the scored `.cact` (sha256 `51f0d3e8ca33…`), the full-precision weights, the three training stages' records (v0, v1, r1), NOTICE and the model card.
- **Cleaned:** the server paths were removed from the weights' metadata and the records.
- **In this repo:** the card, NOTICE and records are kept in [release/AksharSTT-V1/](../release/AksharSTT-V1/).
- **Not released:** v1 itself (run `m1-akshar-v1`), which was r1's starting point.

akshar v2 and akshar-lite v2 are not released. Their weights and records are on Drive under `akshar/phase3-runs/v2-akshar` and `l3-akshar-lite-v2`. Publishing them is the owner's decision. Their NOTICE needs LibriSpeech train-clean-360 beside train-clean-100, as the repository's NOTICE now has.

## Next

- **Svarah:** the last set more than 0.5 behind Shruti (1.22).
  - Common Voice's Indian-accented English is admitted (D26) and is the one source Shruti had that akshar lacks. It has to come from Mozilla Data Collective with the owner's account.
  - NPTEL has about 2,600 h more English lectures than v2 used (v2 took a quarter of each file). Their references become targets as v2's did (D28).
- **Shrutilipi:** about 1,100 h still need Shruti's labels; its scripts are not verbatim (D27).
- **Streaming:** v2's pool and feeder can train the next version from all of Drive again. The fixed data on disk (MUCS, NPTEL's labelled clips, LibriSpeech-100, FLEURS, silence) was heard about 13 times; its share could fall as streamed data grows.
