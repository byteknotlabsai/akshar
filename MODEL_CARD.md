---
language:
- hi
- en
license: apache-2.0
pipeline_tag: automatic-speech-recognition
base_model: Cactus-Compute/whistle
tags:
- speech-recognition
- hindi
- hinglish
- code-switching
- on-device
- quantized
datasets:
- ai4bharat/IndicVoices
- ai4bharat/Kathbath
- ai4bharat/Rural_Women_ASR_v2
- ai4bharat/NPTEL
- google/fleurs
- dianavdavidson/MUCS-Hinglish
- openslr/librispeech_asr
---

# akshar

akshar is an offline speech-to-text model for Hindi, English and Hinglish, small enough for phones and ordinary CPUs. It transcribes on the device, with no internet connection. It writes Hindi in Devanagari and English words in Latin script.

Code, training and evaluation: [github.com/byteknotlabsai/akshar](https://github.com/byteknotlabsai/akshar).

## Model details

| | |
|---|---|
| Architecture | [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle) encoder-decoder, 55 million parameters (8 encoder and 8 decoder layers, width 512) |
| Tokenizer | Whistle's, plus [Shruti](https://huggingface.co/udgamlabs/Shruti)'s 714 Devanagari pieces and a Hindi tag |
| Precision | 4-bit weights (akshar); 2-bit with a 4-bit encoder (akshar-lite) |
| Format | `.cact`, run by the needle engine (Windows, macOS, Linux, Android, iOS, WebAssembly) |
| Languages | Hindi, English, and the two mixed (Hinglish) |
| Licence | Apache-2.0; training data credited in [NOTICE](https://github.com/byteknotlabsai/akshar/blob/main/NOTICE) |

## Files

| File | Model | Size | Mean WER |
|---|---|---|---|
| `akshar-v1-ref.cact` | akshar v1, trained on dataset transcripts (recommended) | 29.5 MB | 22.28 |
| `akshar-v1.cact` | akshar v1 | 29.5 MB | 22.45 |
| `akshar-lite-v0.cact` | akshar-lite v0 | 20.4 MB | 24.03 |

## How to use

```bash
pip install cactus-needle==3.2.8
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1   # the needle package reports usage unless told not to
```

```python
from needle import Whistle

model = Whistle("akshar-v1-ref.cact")
# samples: 16 kHz mono float32 audio. language: "hi" for Hindi and Hinglish, "en" for English.
print(model.transcribe(samples, language="hi")["text"])
```

The repository also has an HTTP API (`python -m akshar.serve`) and a deploy package for servers.

## Output

The model writes the `mixed` style: Hindi in Devanagari, English words in lowercase Latin, numbers as words, no punctuation. For example: `तुम्हारा laptop अच्छा लग रहा है`. Fully Devanagari or fully romanised output is planned as a text layer on top.

## Evaluation

Word error rate (%, lower is better) on the full test sets:

| Model | Monsoon Hindi | FLEURS Hindi | Lahaja | MUCS Hinglish | Svarah | LibriSpeech clean | Silence | Mean |
|---|---|---|---|---|---|---|---|---|
| Shruti, 4-bit (reference) | 20.54 | 20.21 | 26.29 | 35.26 | 22.45 | 5.95 | 0 | 21.78 |
| akshar v1, dataset transcripts | 21.78 | 19.37 | 26.49 | 34.41 | 24.68 | 6.92 | 0 | 22.28 |
| akshar v1 | 22.07 | 19.83 | 26.89 | 34.46 | 24.42 | 7.01 | 0 | 22.45 |
| akshar-lite v0 | 24.79 | 21.78 | 28.18 | 35.06 | 26.45 | 7.91 | 0 | 24.03 |

- All models were scored by the project's harness on the same machine. The settings:
  - Normaliser `nfc-lower-punct2space-v1`.
  - Needle engine 3.3.3 with 5 beams.
  - Language flag `hi` for the first four sets, `en` for Svarah and LibriSpeech.
- Monsoon is scored with OIWER against its lattice references; the other sets with WER.
- Silence counts words written on 100 silent clips. It is not part of the mean.

## Training

- **Start:** Whistle's weights with Shruti's tokenizer.
- **Training:** 4-bit quantisation-aware training of every weight. The loss is cross-entropy on the targets plus distillation from Shruti's output distributions, weight 0.5.
- **Targets:** each dataset's own transcripts, converted to the `mixed` style, where they are verbatim. Elsewhere, Shruti's transcripts.
- **Data:** about 530 hours, every test clip removed by audio and text hash:

| Dataset | Hours | Licence |
|---|---|---|
| Kathbath, Hindi | 135 | CC-BY-4.0 |
| MUCS 2021 Hindi-English, train | 92 | CC-BY-SA-4.0 |
| IndicVoices, Hindi | 83 | CC-BY-4.0 |
| Rural Women ASR v2, Hindi | 80 | CC-BY-4.0 |
| NPTEL lectures, Hindi | 60 | CC-BY-4.0 |
| NPTEL lectures, English | 53 | CC-BY-4.0 |
| LibriSpeech train-clean-100 | 22 | CC-BY-4.0 |
| FLEURS Hindi, train | 3 | CC-BY-4.0 |
| Generated silence and noise | 2 | |

- **Compute:** about 6 hours on one RTX 5080.

## Limitations

- **Indian-accented English** (Svarah) is 2.2 points behind Shruti, and conversational phone speech (Monsoon) 1.2 behind. Both are where the next version adds data.
- **Clip length:** trained on clips up to 30 seconds.
- **Formatting:** no punctuation or digits in the output. Numbers come out as words.
- **Loanwords:** English words in Hindi speech may come out in either script, following each dataset's convention. For example, read speech keeps टाइम in Devanagari.
- **Languages:** Hindi and English only. Other Indian languages are not supported.
- **Untested:** on phones, akshar has not been measured yet. Shruti, the same size, peaks at about 80 MB of RAM on Android.

## Licence and attribution

Apache-2.0. The model starts from Cactus Whistle (Apache-2.0) and learns from Shruti (Apache-2.0). Its training data is listed above, and credited in full in the repository's NOTICE.

## Citation

```bibtex
@software{akshar2026,
  title  = {akshar: offline speech-to-text for Hindi, English and Hinglish},
  author = {{The akshar authors}},
  year   = {2026},
  url    = {https://github.com/byteknotlabsai/akshar}
}
```
