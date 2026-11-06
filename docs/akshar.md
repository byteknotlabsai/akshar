# akshar (Track B)

Phase 3 of the [roadmap](roadmap.md): akshar itself, trained from Whistle rather than from Shruti's weights (D9). This page records how each version was made and how it scored.

**Status (2026-11-06):** akshar v0 and akshar-lite v0 exist, both trained in one night on the RTX 5080 (D25). They are working models, not release candidates; see "Release" below.

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

## Running it

The files are `models/akshar/akshar-v0.cact` and `models/akshar/akshar-lite-v0.cact` on the laptop (not committed), and on Drive under `akshar/phase3-runs/`. Any needle engine runs them like Shruti:

```python
from needle import Whistle
print(Whistle("models/akshar/akshar-v0.cact").transcribe(samples, language="hi")["text"])
```

## Release

akshar v0 is not releasable yet, for two reasons that the legal review of D19 covers:
- Its training data includes MUCS 2021, which is `hold, train`.
- Its teacher, Shruti, was trained on MUCS 2021 and Common Voice (the ledger's teacher table).

A version without them would drop MUCS from the data and replace Shruti's labels with converted dataset transcripts or other teachers' labels, then retrain from Whistle.

## Next

- Train longer on more data: the full Kathbath and NPTEL Hindi sets and 817 h of IndicVoices are ready on Drive; Shrutilipi and Vaani wait for their access forms.
- Target the largest gaps, Monsoon's conversations and Svarah's Indian English.
- Convert dataset transcripts to the `mixed` style, so akshar learns from references and not only from Shruti.
