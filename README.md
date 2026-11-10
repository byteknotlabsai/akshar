# AksharSTT

**Offline speech to text for Hindi, English and Hinglish. One 29.5 MB open model that runs on the device.**

[Website](https://akshar.byteknotlabs.com) · [Live demo](https://akshar.byteknotlabs.com/#demo) · [AksharSTT v4 on Hugging Face](https://huggingface.co/byteknotlabs/AksharSTT-V4) · [AksharSTT Lite v4](https://huggingface.co/byteknotlabs/AksharSTT-Lite-V4) · [Model card](MODEL_CARD.md) · [How each version was made](docs/akshar.md)

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AksharSTT--V4-yellow.svg)](https://huggingface.co/byteknotlabs/AksharSTT-V4)
[![Website](https://img.shields.io/badge/try%20it-akshar.byteknotlabs.com-c93a22.svg)](https://akshar.byteknotlabs.com)

![India speaks Hinglish. AksharSTT writes it down.](docs/assets/aksharstt-v4.png)

AksharSTT turns speech into text on the device itself, with no internet connection. The model file is 29.5 MB at 4 bits, or 20.4 MB for AksharSTT Lite. It runs in the [needle engine](https://huggingface.co/Cactus-Compute/whistle), which embeds in Windows, macOS, Linux, Android, iOS and web apps. Hindi comes out in Devanagari and English words in Latin script, the way people write Hinglish:

```text
जैसे कि आप देख सकते हैं workspace में पांच tabs हैं
```

This repository, akshar, holds everything behind the model: the training code, the data pipeline, the evaluation harness and the records of every version.

## News

- **10 Nov 2026: AksharSTT v4 and AksharSTT Lite v4** are released with open weights under Apache-2.0. v4 averages 20.78 word error rate over six public test sets against Shruti's 21.78, and is lower on every one of the six (same clips, same harness). Lite v4, at 20.4 MB, averages 21.71. Release notes: [GitHub release v4](https://github.com/byteknotlabsai/akshar/releases/tag/v4).
- **7 Nov 2026: AksharSTT v1**, the first public version, on [Hugging Face](https://huggingface.co/byteknotlabs/AksharSTT-V1).

The public versions are v1 and v4. v0, v2 and v3 were internal steps; their results are in the table below.

## Try it

- **In your browser:** [akshar.byteknotlabs.com](https://akshar.byteknotlabs.com). Record a few seconds or drop in an audio file, and get the transcript with a timing for every word. Pick AksharSTT v4, Lite v4, v1 or Shruti, or run them all and compare. Your recording goes to our demo server to be transcribed.
- **On your own machine:** download [AksharSTT v4](https://huggingface.co/byteknotlabs/AksharSTT-V4) (`akshar-v4.cact`, 29.5 MB) or [AksharSTT Lite v4](https://huggingface.co/byteknotlabs/AksharSTT-Lite-V4) (`akshar-lite-v4.cact`, 20.4 MB) and run it offline with the needle engine, as in [Quick start](#quick-start).

## Results

Word error rate (%, lower is better) on the full test sets:

| Model | File | Monsoon Hindi | FLEURS Hindi | Lahaja | MUCS Hinglish | Svarah | LibriSpeech clean | Silence | Mean |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit | 29.5 MB | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| akshar v4, averaged weights | 29.5 MB | **19.35** | 18.41 | 25.40 | 34.18 | 21.48 | 5.86 | 0 | **20.78** |
| **AksharSTT v4** (akshar v4, final weights) | 29.5 MB | 19.47 | 18.45 | 25.45 | **34.07** | **21.41** | **5.82** | 0 | **20.78** |
| akshar v3, averaged weights (internal) | 29.5 MB | 19.70 | **18.27** | **24.87** | 34.09 | 21.85 | 5.88 | 0 | 20.78 |
| akshar v3, final weights (internal) | 29.5 MB | 19.82 | 18.32 | 24.88 | 34.24 | 21.90 | 5.93 | 0 | 20.85 |
| akshar v2 (internal) | 29.5 MB | 20.45 | 18.42 | 25.05 | 34.30 | 23.67 | 6.03 | 0 | 21.32 |
| akshar v1, dataset transcripts (AksharSTT v1) | 29.5 MB | 21.78 | 19.37 | 26.49 | 34.41 | 24.68 | 6.92 | 0 | 22.28 |
| akshar v1, teacher stage (internal) | 29.5 MB | 22.07 | 19.83 | 26.89 | 34.46 | 24.42 | 7.01 | 0 | 22.45 |
| akshar v0 (internal) | 29.5 MB | 23.53 | 20.92 | 27.70 | 34.75 | 24.81 | 6.98 | 0 | 23.11 |
| **AksharSTT Lite v4** (akshar-lite v4) | 20.4 MB | 21.55 | 19.14 | 25.78 | 34.85 | 22.43 | 6.50 | 0 | **21.71** |
| akshar-lite v3 | 20.4 MB | 22.07 | 19.27 | 24.53 | 34.79 | 23.47 | 6.53 | 0 | 21.78 |
| akshar-lite v3, from the averaged weights | 20.4 MB | 21.27 | 19.57 | 24.97 | 34.55 | 23.05 | 6.59 | 0 | 21.67 |
| akshar-lite v2 | 20.4 MB | 22.45 | 19.73 | 25.06 | 34.63 | 25.07 | 6.74 | 0 | 22.28 |
| akshar-lite v0 | 20.4 MB | 24.79 | 21.78 | 28.18 | 35.06 | 26.45 | 7.91 | 0 | 24.03 |

- Every model was scored by our harness, `python -m akshar.eval`, on the same machine. The settings:
  - Normaliser `nfc-lower-punct2space-v1`.
  - Needle engine 3.3.3 with 5 beams.
  - Language flag `hi` for Monsoon, FLEURS, Lahaja and MUCS, and `en` for Svarah and LibriSpeech.
- Monsoon is scored with OIWER against its lattice references; the other sets with WER.
- Silence counts words written on 100 silent clips. It is not part of the mean.
- Test sets, pins and protocol: [docs/eval.md](docs/eval.md).

**The test sets:**
- **Monsoon:** Hindi phone conversations.
- **FLEURS:** read Hindi.
- **Lahaja:** accented Hindi.
- **MUCS:** Hindi-English code-mixing.
- **Svarah:** Indian-accented English.
- **LibriSpeech:** American-accented read English.

How each version was made: [docs/akshar.md](docs/akshar.md).

## Models

| Model | Precision | File | Mean WER | Use it for | Download |
|---|---|---|---|---|---|
| AksharSTT v4 | 4-bit | 29.5 MB | 20.78 | The best accuracy | [byteknotlabs/AksharSTT-V4](https://huggingface.co/byteknotlabs/AksharSTT-V4) |
| AksharSTT Lite v4 | 2-bit, encoder 4-bit | 20.4 MB | 21.71 | Tight memory budgets; about one point of WER worse | [byteknotlabs/AksharSTT-Lite-V4](https://huggingface.co/byteknotlabs/AksharSTT-Lite-V4) |
| AksharSTT v1 | 4-bit | 29.5 MB | 22.28 | The first release | [byteknotlabs/AksharSTT-V1](https://huggingface.co/byteknotlabs/AksharSTT-V1) |

This repository holds the code, never the weights. The model card is [MODEL_CARD.md](MODEL_CARD.md); the cards, NOTICE files and training records of each release are in [release/](release/).

## Quick start

akshar needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Training needs Linux with an NVIDIA GPU; running a model does not.

```bash
git clone https://github.com/byteknotlabsai/akshar
cd akshar
uv sync
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1   # the needle package reports usage unless told not to
```

Transcribe a file with AksharSTT v4 from Hugging Face:

```python
from huggingface_hub import hf_hub_download
from needle import Whistle
from akshar.eval import audio

model = Whistle(hf_hub_download("byteknotlabs/AksharSTT-V4", "akshar-v4.cact"))
samples = audio.load("speech.wav")  # any WAV or FLAC, resampled to 16 kHz mono
print(model.transcribe(samples, language="hi")["text"])  # "hi" for Hindi and Hinglish, "en" for English
```

Two ready-made ways to use it:

- **HTTP API for apps:** `uv run python -m akshar.serve`, then POST a recording to `/v1/transcribe`. To run it on another machine, `uv run python scripts/build_deploy.py --zip` packs it into one folder. See [deploy/README.md](deploy/README.md).
- **Local page:** `uv run python -m akshar.ui` opens a page on 127.0.0.1. Record or upload audio and compare the models, or browse every test clip and its errors.

## Output styles

The model writes the `mixed` style: Hindi in Devanagari, English words in lowercase Latin, numbers as words, no punctuation. It is the only style that records which words were spoken in English, so the other two are rendered from it by a small text layer:

| Style | Example |
|---|---|
| `mixed` | तुम्हारा laptop अच्छा लग रहा है |
| `devanagari` | तुम्हारा लैपटॉप अच्छा लग रहा है |
| `roman` | tumhara laptop accha lag raha hai |

The text layer is still to come ([roadmap](docs/roadmap.md), Phase 5). Reasons: decisions D3 and D11.

## How akshar is made

- **Base model:** [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle), an encoder-decoder speech model with 55 million parameters. akshar gives it Shruti's Devanagari tokenizer and trains every weight.
- **Precision:** 4-bit quantisation-aware training throughout, so the 29.5 MB file is what was trained, not a compressed afterthought.
- **Targets:** each dataset's own human transcripts, converted to the `mixed` style, wherever they are verbatim (`akshar.data.convert`, D27). Elsewhere, Shruti's transcripts. Distillation from Shruti's output distributions throughout.
- **akshar-lite:** made from akshar with the 2-bit recipe found in [Phase 2](docs/two-bit.md).
- **Compute:** one RTX 5080, in night windows; v2 took about 6.5 GPU-hours, v3 about 9 and v4 about 6. Free compute only.
- **Data on a small disk:** since v2 the data streams from Google Drive while the model trains (`akshar.train.feed`), so v3 learned from about 3,900 hours on a server that holds about 500 (D28).
- **Indian English:** v3 added Common Voice's South Asian English and more NPTEL lectures; v4 trains 40% of that English under the Hindi tag so that English stays English in the default Hindi mode (D30).
- **English words in Latin:** a curated list of 2,900 English words, brands and place names with their Devanagari spellings ([data/english-words](data/english-words/), D32) decides what the training targets write in Latin.

## Training data

AksharSTT v4 learned from about 3,900 hours, counting each clip of its run once (v2 from about 3,000, v1 from about 530):

| Source | Hours | Licence |
|---|---|---|
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), English lectures | 1,600 | CC-BY-4.0 |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), Hindi | 723 | CC-BY-4.0 |
| [LibriSpeech](https://www.openslr.org/12/), train-clean-360 and part of train-clean-100 | 374 | CC-BY-4.0 |
| [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part), transcribed Hindi | 352 | CC-BY-4.0 |
| [Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi), Hindi (263 h transcribed by Shruti, 83 h by the internal v2) | 346 | CC-BY-4.0 |
| [Common Voice](https://datacollective.mozillafoundation.org/) Scripted Speech 26.0, South Asian English | 158 | CC0 |
| [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath), Hindi | 133 | CC-BY-4.0 |
| [MUCS 2021](https://www.openslr.org/104/), Hindi-English | 92 | CC-BY-SA-4.0 |
| [Rural Women ASR v2](https://huggingface.co/datasets/ai4bharat/Rural_Women_ASR_v2), Hindi | 76 | CC-BY-4.0 |
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), Hindi lectures | 60 | CC-BY-4.0 |
| [Common Voice](https://datacollective.mozillafoundation.org/) 27.0, Hindi | 9 | CC0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), Hindi train | 3 | CC-BY-4.0 |
| Silence and noise, generated | 2 | |

v4 trained on the repaired transcripts of D29 and the English-word list of D32. The full per-source record of its run is [release/AksharSTT-V4/training/v4.json](release/AksharSTT-V4/training/v4.json).

**Rules:**
- No dataset is used without an entry in the licence ledger, [data/LICENSES.md](data/LICENSES.md). Every source is credited in [NOTICE](NOTICE).
- Every test clip is removed from the training data, by audio hash and by text hash (D17).

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | An evaluation harness that reproduces Shruti's published scores | Done ([eval.md](docs/eval.md)) |
| 1 | Train Whistle's weights and export 4-bit and 2-bit models | Done ([toolchain.md](docs/toolchain.md)) |
| 2 | The 2-bit recipe for akshar-lite | Done ([two-bit.md](docs/two-bit.md)) |
| 3 | akshar from Whistle | Done: v3 and v4 beat Shruti on every test set ([akshar.md](docs/akshar.md)) |
| 4 | Windows, Mac and phones | Next |
| 5 | Output styles (the text layer) | Planned |
| 6 | Release | AksharSTT v4 and Lite v4 released on 10 Nov 2026; apps follow Phases 4 and 5 |

**The release bar:**
- **akshar:** a lower mean WER than Shruti, with no test set more than 0.5 worse.
- **akshar-lite:** within 2 points of akshar.

Next: Phase 4, the Windows, Mac and phone apps, and Phase 5, the text layer. Details: [docs/roadmap.md](docs/roadmap.md).

## Repository layout

| Path | What it is |
|---|---|
| [akshar/eval/](akshar/eval/) | The evaluation harness: test-set download, decoding in the needle engine, scoring |
| [akshar/data/](akshar/data/) | Training data: building, Drive staging, Shruti labels, transcript conversion |
| [akshar/model/](akshar/model/), [akshar/train/](akshar/train/) | Checkpoints, quantisation, `.cact` export; quantisation-aware distillation |
| [akshar/serve/](akshar/serve/), [deploy/](deploy/) | The HTTP API and its deploy package |
| [akshar/ui/](akshar/ui/) | The local page for trying models and browsing results |
| [scripts/](scripts/) | GPU-server, deploy and daily Google Drive backup scripts |
| [docs/](docs/) | Roadmap, decisions (D1 to D33), harness, results for each phase |
| [data/LICENSES.md](data/LICENSES.md) | The dataset licence ledger |
| [MODEL_CARD.md](MODEL_CARD.md) | The model card of AksharSTT v4, as on Hugging Face |
| [release/](release/) | The card, NOTICE and training records of each Hugging Face release |

New to the code? Start with [docs/roadmap.md](docs/roadmap.md) and [docs/decisions.md](docs/decisions.md); [docs/setup-ubuntu.md](docs/setup-ubuntu.md) sets up a training machine.

## License

The code is under the [Apache License 2.0](LICENSE). The model weights are released under Apache-2.0 too, with every training dataset credited in [NOTICE](NOTICE) as its licence requires.

## Credits

akshar builds on [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle) and the needle engine by Cactus Compute (Apache-2.0). Its teacher is [Shruti](https://huggingface.co/udgamlabs/Shruti) by Udgam Labs (Apache-2.0). Its training data comes from AI4Bharat, Google (FLEURS), the MUCS 2021 challenge and LibriSpeech; [NOTICE](NOTICE) lists every dataset.

## Citation

```bibtex
@software{aksharstt2026,
  title   = {AksharSTT: offline speech-to-text for Hindi, English and Hinglish},
  author  = {{ByteKnot Labs}},
  year    = {2026},
  version = {v4},
  url     = {https://huggingface.co/byteknotlabs/AksharSTT-V4},
  note    = {Code: https://github.com/byteknotlabsai/akshar}
}
```
