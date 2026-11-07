# akshar

**Offline speech-to-text for Hindi, English and Hinglish, small enough for phones and ordinary CPUs.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

akshar turns speech into text on the device itself, with no internet connection. The model file is 29.5 MB at 4 bits, or 20.4 MB for akshar-lite. It runs in the [needle engine](https://huggingface.co/Cactus-Compute/whistle), which embeds in Windows, macOS, Linux, Android, iOS and web apps. Hindi comes out in Devanagari and English words in Latin script, the way people write Hinglish:

```text
जैसे कि आप देख सकते हैं workspace में पाँच tabs हैं
```

**Status (2026-11-07):** akshar v1 is trained and tested. It beats [Shruti](https://huggingface.co/udgamlabs/Shruti), the open model it is measured against, on FLEURS Hindi and MUCS Hinglish, and trails it by 0.50 points of average word error rate. The weights will be published on Hugging Face. See [Roadmap](#roadmap) for what comes next.

## Results

Word error rate (%, lower is better) on the full test sets:

| Model | File | Monsoon Hindi | FLEURS Hindi | Lahaja | MUCS Hinglish | Svarah | LibriSpeech clean | Silence | Mean |
|---|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit | 29.5 MB | **20.54** | 20.21 | **26.29** | 35.26 | **22.45** | **5.95** | 0 | **21.78** |
| akshar v1, dataset transcripts | 29.5 MB | 21.78 | **19.37** | 26.49 | **34.41** | 24.68 | 6.92 | 0 | 22.28 |
| akshar v1 | 29.5 MB | 22.07 | 19.83 | 26.89 | 34.46 | 24.42 | 7.01 | 0 | 22.45 |
| akshar v0 | 29.5 MB | 23.53 | 20.92 | 27.70 | 34.75 | 24.81 | 6.98 | 0 | 23.11 |
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

| Model | Precision | File | Use it for |
|---|---|---|---|
| akshar | 4-bit | 29.5 MB | The best accuracy |
| akshar-lite | 2-bit, encoder 4-bit | 20.4 MB | Tight memory budgets; about one point of WER worse |

The weights will be on Hugging Face; this repository holds the code, never the weights. The model card is [MODEL_CARD.md](MODEL_CARD.md).

## Quick start

akshar needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Training needs Linux with an NVIDIA GPU; running a model does not.

```bash
git clone https://github.com/byteknotlabsai/akshar
cd akshar
uv sync
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1   # the needle package reports usage unless told not to
```

Transcribe a file with a downloaded model:

```python
from needle import Whistle
from akshar.eval import audio

model = Whistle("models/akshar/akshar-v1.cact")
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
- **Compute:** one RTX 5080, in night windows; v1 took about 6 GPU-hours. Free compute only.

## Training data

akshar v1 learned from about 530 hours:

| Source | Hours | Licence |
|---|---|---|
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), Hindi | 83 | CC-BY-4.0 |
| [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath), Hindi | 135 | CC-BY-4.0 |
| [MUCS 2021](https://www.openslr.org/104/), Hindi-English | 92 | CC-BY-SA-4.0 |
| [Rural Women ASR v2](https://huggingface.co/datasets/ai4bharat/Rural_Women_ASR_v2), Hindi | 80 | CC-BY-4.0 |
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), Hindi lectures | 60 | CC-BY-4.0 |
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), English lectures | 53 | CC-BY-4.0 |
| [LibriSpeech](https://www.openslr.org/12/), train-clean-100 | 22 | CC-BY-4.0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), Hindi train | 3 | CC-BY-4.0 |
| Silence and noise, generated | 2 | |

Ready for the next version: [Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi) (1,374 h), [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) (644 h) and the rest of IndicVoices (about 735 h).

**Rules:**
- No dataset is used without an entry in the licence ledger, [data/LICENSES.md](data/LICENSES.md). Every source is credited in [NOTICE](NOTICE).
- Every test clip is removed from the training data, by audio hash and by text hash (D17).

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | An evaluation harness that reproduces Shruti's published scores | Done ([eval.md](docs/eval.md)) |
| 1 | Train Whistle's weights and export 4-bit and 2-bit models | Done ([toolchain.md](docs/toolchain.md)) |
| 2 | The 2-bit recipe for akshar-lite | Done ([two-bit.md](docs/two-bit.md)) |
| 3 | akshar from Whistle | In progress: v1 trails Shruti by 0.50 ([akshar.md](docs/akshar.md)) |
| 4 | Windows, Mac and phones | Next |
| 5 | Output styles (the text layer) | Planned |
| 6 | Release | Planned |

**The release bar:**
- **akshar:** a lower mean WER than Shruti, with no test set more than 0.5 worse.
- **akshar-lite:** within 2 points of akshar.

Next: train on the 2,750 new hours above, and close the English gap (Svarah, LibriSpeech). Details: [docs/roadmap.md](docs/roadmap.md).

## Repository layout

| Path | What it is |
|---|---|
| [akshar/eval/](akshar/eval/) | The evaluation harness: test-set download, decoding in the needle engine, scoring |
| [akshar/data/](akshar/data/) | Training data: building, Drive staging, Shruti labels, transcript conversion |
| [akshar/model/](akshar/model/), [akshar/train/](akshar/train/) | Checkpoints, quantisation, `.cact` export; quantisation-aware distillation |
| [akshar/serve/](akshar/serve/), [deploy/](deploy/) | The HTTP API and its deploy package |
| [akshar/ui/](akshar/ui/) | The local page for trying models and browsing results |
| [scripts/](scripts/) | GPU-server, deploy and daily Google Drive backup scripts |
| [docs/](docs/) | Roadmap, decisions (D1 to D27), harness, results for each phase |
| [data/LICENSES.md](data/LICENSES.md) | The dataset licence ledger |
| [MODEL_CARD.md](MODEL_CARD.md) | The model card for the Hugging Face release |

New to the code? Start with [docs/roadmap.md](docs/roadmap.md) and [docs/decisions.md](docs/decisions.md); [docs/setup-ubuntu.md](docs/setup-ubuntu.md) sets up a training machine.

## License

The code is under the [Apache License 2.0](LICENSE). The model weights will be released under Apache-2.0 too, with every training dataset credited in [NOTICE](NOTICE) as its licence requires.

## Credits

akshar builds on [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle) and the needle engine by Cactus Compute (Apache-2.0). Its teacher is [Shruti](https://huggingface.co/udgamlabs/Shruti) by Udgam Labs (Apache-2.0). Its training data comes from AI4Bharat, Google (FLEURS), the MUCS 2021 challenge and LibriSpeech; [NOTICE](NOTICE) lists every dataset.

## Citation

```bibtex
@software{akshar2026,
  title  = {akshar: offline speech-to-text for Hindi, English and Hinglish},
  author = {{The akshar authors}},
  year   = {2026},
  url    = {https://github.com/byteknotlabsai/akshar}
}
```
