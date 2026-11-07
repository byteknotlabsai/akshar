# akshar (Track B)

Phase 3 of the [roadmap](roadmap.md): akshar itself, trained from Whistle rather than from Shruti's weights (D9). This page records how each version was made and how it scored.

**Status (2026-11-07):** akshar v1 trails Shruti by 0.67 points of mean WER on the full test sets, half of v0's gap, and beats it on FLEURS and MUCS. akshar v0 and akshar-lite v0 came first, each trained in one night on the RTX 5080 (D25). None is a release candidate yet; see "Release" below.

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

## Running it

The files are `models/akshar/akshar-v1.cact`, `akshar-v0.cact` and `akshar-lite-v0.cact` on the laptop (not committed), and on Drive under `akshar/phase3-runs/`. Any needle engine runs them like Shruti:

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

What stands between akshar and a release is now accuracy. The roadmap's bar is a lower mean than Shruti's with no set more than 0.5 worse, and v1 is still 0.6 to 2.0 behind on four sets.

On 7 November the owner released r1 as **AksharSTT v1**, before that bar is met: [huggingface.co/byteknotlabs/AksharSTT-V1](https://huggingface.co/byteknotlabs/AksharSTT-V1), Apache-2.0.
- **Contents:** the scored `.cact` (sha256 `51f0d3e8ca33…`), the full-precision weights, the three training stages' records (v0, v1, r1), NOTICE and the model card.
- **Cleaned:** the server paths were removed from the weights' metadata and the records.
- **In this repo:** the card, NOTICE and records are kept in [release/AksharSTT-V1/](../release/AksharSTT-V1/).
- **Not released:** v1 itself (run `m1-akshar-v1`), which was r1's starting point.

## Next

- **v2:** train on all the data Drive now holds, with dataset transcripts as targets (D27).
  - IndicVoices: 817 h.
  - Vaani's transcribed Hindi: 644 h.
  - Shrutilipi: 1,374 h, of which 267 h are labelled by Shruti so far.
  - Everything v1 used.
  - That is about 3,300 h, far more than the server's disk holds. v2 needs slices that rotate through the disk during a run, and a compile cache so that each restart does not cost 12 to 15 minutes.
- **English:** Svarah (2.2 behind) and LibriSpeech (1.0 behind) are now the largest gaps.
  - Common Voice's Indian-accented English is admitted (D26), but has to come from Mozilla Data Collective with the owner's account; Hugging Face no longer has it.
  - NPTEL English's transcripts could become targets once digits are spelled out.
- **akshar-lite v1:** made from the best model with D24's recipe, about 25 minutes of GPU.
