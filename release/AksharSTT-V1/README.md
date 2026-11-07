---
license: apache-2.0
language:
- hi
- en
pipeline_tag: automatic-speech-recognition
base_model: Cactus-Compute/whistle
datasets:
- ai4bharat/IndicVoices
- ai4bharat/Kathbath
- ai4bharat/Rural_Women_ASR_v2
- google/fleurs
- ai4bharat/NPTEL
- openslr/librispeech_asr
- dianavdavidson/MUCS-Hinglish
tags:
- speech-recognition
- hindi
- hinglish
- code-switching
- on-device
- cactus
---

# AksharSTT v1

AksharSTT is offline speech-to-text for Hindi, English and Hinglish, made to run on ordinary CPUs and phones. Version 1 is a single 29.5 MB file with 4-bit weights, run by Cactus's needle engine. It starts from [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle) and was trained with [Shruti](https://huggingface.co/udgamlabs/Shruti) as its teacher.

Its word error rate averages 22.28% over six test sets, half a point behind Shruti's 21.78%. It is ahead of Shruti on FLEURS Hindi and on MUCS Hinglish, and furthest behind on Indian-accented English. The training code, the data pipeline and the evaluation harness are at [github.com/byteknotlabsai/akshar](https://github.com/byteknotlabsai/akshar).

## Output

The model writes one style: Hindi in Devanagari, English words in lowercase Latin letters, numbers as words, and no punctuation. On a Hinglish clip from the MUCS test set it writes:

```
जैसे कि आप देख सकते हैं workspace में पांच tabs हैं
```

Common English loanwords can come out in either script. Punctuation, digits and other scripts are left to the application.

## Quick start

```bash
pip install cactus-needle==3.2.8 huggingface_hub
```

```python
import os
os.environ.setdefault("NEEDLE_TELEMETRY", "0")  # needle reports usage unless these are set
os.environ.setdefault("DO_NOT_TRACK", "1")

from huggingface_hub import hf_hub_download
from needle import Whistle

model = Whistle(hf_hub_download("byteknotlabs/AksharSTT-V1", "akshar-v1.cact"))
print(model.transcribe("speech.wav", language="hi")["text"])
```

Use `language="hi"` for Hindi and Hinglish and `language="en"` for English. `transcribe` takes a WAV file or 16 kHz mono float samples in [-1, 1], of any length. With `word_timestamps=True` it also returns each word's start, end and probability.

## Accuracy

Word error rate in percent on the full test sets, lower is better. Both models were scored on the same machine (AMD Ryzen 9 7900X) by akshar's harness: normaliser `nfc-lower-punct2space-v1`, needle engine build 3.3.3 (cactus-needle 3.2.8) with its default 5 beams, language `hi` for the Hindi and Hinglish sets and `en` for the English ones.

| Test set | Clips | Speech | Shruti, 4-bit | AksharSTT v1 |
|---|---|---|---|---|
| [Monsoon](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN), Hindi | 753 | phone conversations | 20.54 | 21.78 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), Hindi test | 418 | read sentences | 20.21 | 19.37 |
| [Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja) | 6,152 | Hindi in many accents | 26.29 | 26.49 |
| [MUCS 2021](https://www.openslr.org/104), Hindi-English test | 3,136 | technical tutorials, code-mixed | 35.26 | 34.41 |
| [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | 6,656 | Indian-accented English | 22.45 | 24.68 |
| [LibriSpeech](https://www.openslr.org/12) test-clean | 2,620 | read English | 5.95 | 6.92 |
| Mean of the six | | | 21.78 | 22.28 |

Monsoon is scored with OIWER, which accepts any of the spellings its references list for a word, Devanagari or Latin; the other sets with plain WER. On 100 clips of silence and noise neither model writes a word.

## Training

AksharSTT v1 has Whistle's architecture: 8 encoder and 8 decoder layers of width 512, about 55 million parameters. Whistle's tokenizer was extended with the 714 Devanagari pieces and the `hi` language tag that Shruti added, each new embedding starting as Shruti's report describes. Training then ran in three stages, about 11.5 hours on one RTX 5080. Every stage used 4-bit quantisation-aware training with 8-bit activations, and distillation from Shruti as released, weighted 0.5 against the targets.

1. 56,000 updates at a learning rate of 8e-4, with Shruti's transcripts as targets, on up to 364 hours of audio.
2. 64,000 more updates at 4e-4, still on Shruti's transcripts, on the data below (524 hours of it for training).
3. 2,400 more at 1e-4 with each dataset's own transcripts as targets where those are verbatim (IndicVoices, Kathbath, Rural Women, FLEURS and LibriSpeech), converted to the output style above. MUCS and NPTEL kept Shruti's transcripts. This stage lowered the mean word error rate from 22.45 to 22.28.

An update is about 480 seconds of padded audio. The records of the three stages, with their settings and validation losses, are in `training/`.

| Training data | Licence | Hours |
|---|---|---|
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), Hindi | CC-BY-4.0 | 83 |
| [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath), Hindi | CC-BY-4.0 | 135 |
| [Rural Women ASR v2](https://huggingface.co/datasets/ai4bharat/Rural_Women_ASR_v2), Hindi | CC-BY-4.0 | 80 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), Hindi train | CC-BY-4.0 | 3 |
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), English lectures | CC-BY-4.0 | 53 |
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), Hindi lectures | CC-BY-4.0 | 60 |
| [LibriSpeech](https://www.openslr.org/12), part of train-clean-100 | CC-BY-4.0 | 22 |
| [MUCS 2021](https://www.openslr.org/104) Hindi-English, train | CC-BY-SA-4.0 | 92 |
| Silence and noise generated by akshar's code | | 2 |

Clips over 30 seconds were dropped, and so was every clip that matches a test clip by its audio, or by its text when the text has three or more words. 1% of the rest was held out for validation.

## Limitations

AksharSTT v1 is behind Shruti on Indian-accented English (Svarah, by 2.2 points), on phone conversations (Monsoon, by 1.2) and on LibriSpeech (by 1.0). It was trained and tested only on Hindi and English; Whistle's other language tags are still in its vocabulary, untested. Its scores were measured with engine build 3.3.3; other builds can move them a little.

## Files

- `akshar-v1.cact`: the model for the needle engine, 4-bit, 29.5 MB. This is the file the scores above were measured with.
- `akshar-v1.safetensors`: the full-precision weights, in akshar's checkpoint format (config and tokenizer in its metadata), for further training with the [akshar](https://github.com/byteknotlabsai/akshar) code.
- `training/`: the training records of the three stages.
- `NOTICE`: what AksharSTT v1 is built from, with licences and links.

## Licence

Apache-2.0. Whistle and Shruti are Apache-2.0 too. The training datasets are CC-BY-4.0, except MUCS 2021, which is CC-BY-SA-4.0; `NOTICE` credits each of them.
