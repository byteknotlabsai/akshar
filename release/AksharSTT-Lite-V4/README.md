---
license: apache-2.0
language:
- hi
- en
pipeline_tag: automatic-speech-recognition
base_model: byteknotlabs/AksharSTT-V4
base_model_relation: quantized
datasets:
- ai4bharat/IndicVoices
- ARTPARK-IISc/Vaani-transcription-part
- ai4bharat/Shrutilipi
- ai4bharat/Kathbath
- ai4bharat/Rural_Women_ASR_v2
- ai4bharat/NPTEL
- google/fleurs
- openslr/librispeech_asr
- dianavdavidson/MUCS-Hinglish
tags:
- speech-recognition
- hindi
- hinglish
- code-switching
- indian-english
- on-device
- offline
- cactus
- needle
- quantized
- 2-bit
model-index:
- name: AksharSTT Lite v4
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech recognition
    dataset:
      name: Monsoon Hindi (public split)
      type: VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN
      split: test
      revision: a4b7375e9adea16c16100295976cdb0aabd9fb6d
    metrics:
    - type: wer
      value: 21.55
      name: OIWER
  - task:
      type: automatic-speech-recognition
      name: Speech recognition
    dataset:
      name: FLEURS Hindi
      type: google/fleurs
      config: hi_in
      split: test
      revision: 70bb2e84b976b7e960aa89f1c648e09c59f894dd
    metrics:
    - type: wer
      value: 19.14
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Speech recognition
    dataset:
      name: Lahaja
      type: ai4bharat/Lahaja
      split: test
      revision: d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7
    metrics:
    - type: wer
      value: 25.78
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Speech recognition
    dataset:
      name: MUCS 2021 Hindi-English
      type: dianavdavidson/MUCS-Hinglish
      split: test
      revision: 3c66ee127a7b0e0a746d4df1002858567135edc1
    metrics:
    - type: wer
      value: 34.85
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Speech recognition
    dataset:
      name: Svarah
      type: ai4bharat/Svarah
      split: test
      revision: ebbf7777fe771490696a3f7b007097606fa8c924
    metrics:
    - type: wer
      value: 22.43
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Speech recognition
    dataset:
      name: LibriSpeech test-clean
      type: openslr/librispeech_asr
      config: clean
      split: test
      revision: 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1
    metrics:
    - type: wer
      value: 6.50
      name: WER
---

# AksharSTT Lite v4

Offline speech-to-text for Hindi, English and Hinglish, in a 20.4 MB file for phones and devices with little memory.

[Website](https://akshar.byteknotlabs.com) · [Live demo](https://akshar.byteknotlabs.com) · [GitHub](https://github.com/byteknotlabsai/akshar) · [AksharSTT v4](https://huggingface.co/byteknotlabs/AksharSTT-V4) · [AksharSTT v1](https://huggingface.co/byteknotlabs/AksharSTT-V1)

AksharSTT Lite v4 is [AksharSTT v4](https://huggingface.co/byteknotlabs/AksharSTT-V4) retrained at lower precision: most of its decoder is 2-bit, and its encoder stays 4-bit. It is 9.1 MB smaller than AksharSTT v4 and 0.93 points of mean word error rate behind it. Like AksharSTT v4, it runs in Cactus's needle engine on the device itself.

## Highlights

- 20.4 MB, against 29.5 MB for AksharSTT v4 and for Shruti at 4 bits.
- Mean word error rate 21.71 over six test sets: 0.93 behind AksharSTT v4 and 0.07 below Shruti at 4 bits (21.78), on the same clips, scored by the same harness.
- Indian-accented English (Svarah): 22.43, level with Shruti (22.45) and 2.25 better than AksharSTT v1.
- Lower than AksharSTT v1 on five of the six sets and on the mean (21.71 against 22.28).
- On 100 clips of silence and faint noise it writes nothing.

All scores are on the full test sets with the settings given under [Accuracy](#accuracy).

## Model details

| | |
|---|---|
| Architecture | Whistle encoder-decoder: 8 encoder and 8 decoder layers of width 512, about 55 million parameters |
| Vocabulary | Whistle's, plus Shruti's 714 Devanagari pieces and the `hi` language tag |
| Weights | The decoder's matrices at 2 bits; the encoder, the embedding table and the routing matrices at 4 bits (89 quantised tensors at 2 bits, 63 at 4 bits), trained at that precision |
| Input | A WAV file, or 16 kHz mono float samples in [-1, 1], of any length |
| Output | Text in one style (see [Output](#output)), with optional word timings and confidences |
| Languages | Hindi, English and code-mixed Hindi-English, selected with the language flag `hi` or `en` |
| Runtime | The needle engine (`cactus-needle` on PyPI), which embeds in Windows, macOS, Linux, Android, iOS and web apps |
| Licence | Apache-2.0 |

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

model = Whistle(hf_hub_download("byteknotlabs/AksharSTT-Lite-V4", "akshar-lite-v4.cact"))
print(model.transcribe("speech.wav", language="hi")["text"])
```

Use `language="hi"` for Hindi and Hinglish, and `language="en"` when you know the speech is English (see [Language flag](#language-flag)). `transcribe` takes a WAV file or 16 kHz mono float samples in [-1, 1], of any length.

With `word_timestamps=True` the result also lists every word with its start and end in seconds and the model's probability for it:

```python
result = model.transcribe("speech.wav", language="hi", word_timestamps=True)
for w in result["words"]:
    print(w["word"], w["start"], w["end"], w["probability"])
```

## Output

The model writes one style: Hindi in Devanagari, English words in lowercase Latin letters, numbers as words, and no punctuation. On two test clips it writes:

| Clip | AksharSTT Lite v4 writes |
|---|---|
| MUCS 2021, a software tutorial | जैसे कि आप देख सकते हैं workspace में पाँच tabs हैं |
| Monsoon, a phone call | के लिए half ticket है maharashtra में |

The training transcripts write English words, brands and place names in Latin when they are on a curated list of 2,900 words ([data/english-words](https://github.com/byteknotlabsai/akshar/tree/main/data/english-words)), so the model does the same; other loanwords can come out in either script. Punctuation, digits and other scripts are left to the application.

## Accuracy

Word error rate in percent on the full test sets, lower is better.

| Test set | Clips | Speech | Shruti, 4-bit | AksharSTT v1 | AksharSTT v4 | AksharSTT Lite v4 |
|---|---|---|---|---|---|---|
| [Monsoon](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN), Hindi | 753 | phone conversations | 20.54 | 21.78 | 19.47 | 21.55 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), Hindi test | 418 | read sentences | 20.21 | 19.37 | 18.45 | 19.14 |
| [Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja) | 6,152 | Hindi in many accents | 26.29 | 26.49 | 25.45 | 25.78 |
| [MUCS 2021](https://www.openslr.org/104), Hindi-English test | 3,136 | technical tutorials, code-mixed | 35.26 | 34.41 | 34.07 | 34.85 |
| [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | 6,656 | Indian-accented English | 22.45 | 24.68 | 21.41 | 22.43 |
| [LibriSpeech](https://www.openslr.org/12) test-clean | 2,620 | read English | 5.95 | 6.92 | 5.82 | 6.50 |
| Mean of the six | | | 21.78 | 22.28 | 20.78 | 21.71 |
| File size | | | 29.5 MB | 29.5 MB | 29.5 MB | 20.4 MB |

All four models were scored on the same clips and the same machine (AMD Ryzen 9 7900X) by akshar's harness, `python -m akshar.eval`: normaliser `nfc-lower-punct2space-v1`, needle engine build 3.3.3 (cactus-needle 3.2.8) with its default 5 beams, language flag `hi` for Monsoon, FLEURS, Lahaja and MUCS and `en` for Svarah and LibriSpeech. Monsoon is scored with OIWER, which accepts any of the spellings its references list for a word, Devanagari or Latin; the other sets with plain WER. On 100 clips of silence and faint noise none of the four models writes a word. The protocol and the pinned test-set revisions are in [docs/eval.md](https://github.com/byteknotlabsai/akshar/blob/main/docs/eval.md).

Against Shruti at 4 bits, AksharSTT Lite v4 is ahead on FLEURS (by 1.07), Lahaja (by 0.51) and MUCS (by 0.41), level on Svarah, and behind on Monsoon (by 1.01) and LibriSpeech (by 0.55).

### Language flag

The flag matters for English speech. With `hi`, AksharSTT v4 scores 26.97 on Svarah; with `en`, 21.41 (same clips, same harness and settings otherwise). The lite model was not scored with `hi` on Svarah. Apps should pass `en` when they know the speech is English, and `hi` for Hindi and Hinglish.

## Training

### How it was made

- **Start:** AksharSTT v4's final full-precision weights, the ones published as `akshar-v4.safetensors` in [byteknotlabs/AksharSTT-V4](https://huggingface.co/byteknotlabs/AksharSTT-V4).
- **Precision:** quantisation-aware training at the target widths: Cactus's 2-bit mix (embedding table and routing matrices at 4 bits, the other matrices at 2 bits) with the encoder raised to 4 bits.
- **Teacher:** AksharSTT v4 as released (4-bit). The loss is cross-entropy on the targets plus distillation from the teacher at weight 0.5, temperature 1.0.
- **Run:** 3,600 updates at a learning rate of 4e-4, about 18 minutes on one RTX 5080. Peak GPU memory 4.7 GB. An update is about 480 seconds of padded audio.
- **Validation cross-entropy** on 2,691 held-out clips: 0.3309 just after the switch to 2 bits, 0.2339 at the end (AksharSTT v4: 0.2111).
- **Data and targets:** the same as AksharSTT v4's, from the same stream and the same 230 hours kept on disk. This run drew 406 hours from the stream.

AksharSTT v4 itself was trained from [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle), with [Shruti](https://huggingface.co/udgamlabs/Shruti) as its teacher and no weights taken from Shruti. Its card describes the lineage, the changes in v4 and the data in full. The record of this run is `training/lite-v4.json`.

### Data

AksharSTT Lite v4 carries what AksharSTT v4 learned, so its data is v4's:

| Source | Licence | Hours in AksharSTT v4's run | Hours in this run |
|---|---|---|---|
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), English lectures | CC-BY-4.0 | 1,600 | 252 |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), Hindi | CC-BY-4.0 | 723 | 43 |
| [LibriSpeech](https://www.openslr.org/12), train-clean-360 and part of train-clean-100 | CC-BY-4.0 | 374 | 52 |
| [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part), transcribed Hindi | CC-BY-4.0 | 352 | 29 |
| [Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi), Hindi | CC-BY-4.0 | 346 | 27 |
| [Common Voice](https://datacollective.mozillafoundation.org/) Scripted Speech 26.0, South Asian English | CC0 | 158 | 46 |
| [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath), Hindi | CC-BY-4.0 | 133 | 14 |
| [MUCS 2021](https://www.openslr.org/104), Hindi-English train | CC-BY-SA-4.0 | 92 | 92 |
| [Rural Women ASR v2](https://huggingface.co/datasets/ai4bharat/Rural_Women_ASR_v2), Hindi | CC-BY-4.0 | 76 | 9 |
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), Hindi lectures | CC-BY-4.0 | 60 | 60 |
| [Common Voice](https://datacollective.mozillafoundation.org/) 27.0, Hindi | CC0 | 9 | 9 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), Hindi train | CC-BY-4.0 | 3 | 3 |
| Silence and noise, generated by akshar's code | | 2 | 2 |

Hours count each clip once: about 3,930 in AksharSTT v4's run and about 640 in this one. Every training clip that matches a test clip by its audio, or by its text when the text has three or more words, was removed.

## Intended use

- Transcribing Hindi, English and Hinglish speech without a network where memory or download size is tight: phones, small devices and apps that bundle the model.
- Apps that must keep audio on the device.

Where size allows, [AksharSTT v4](https://huggingface.co/byteknotlabs/AksharSTT-V4) is more accurate on every test set.

Out of scope: languages other than Hindi and English, punctuated or formatted text, translation, and telling speakers apart.

## Limitations

- **Against AksharSTT v4:** 0.93 points of mean WER behind, most on phone conversations (Monsoon, by 2.08) and Indian English (Svarah, by 1.02).
- **Against Shruti at 4 bits:** behind on Monsoon (by 1.01) and LibriSpeech (by 0.55).
- **Hinglish tutorials (MUCS):** 34 to 35 WER for every model in the table. MUCS references are the narrators' scripts rather than verbatim transcripts, and they write the same English word in either script, so part of that error is in the references.
- **Accented Hindi (Lahaja):** 25.78, 1.25 worse than the internal lite v3, the same pattern as AksharSTT v4 (see its card).
- **Script on FLEURS and Lahaja:** their references write English loanwords and place names in Devanagari, so a correct word written in Latin counts as an error on those two sets.
- **Numbers and punctuation:** numbers come out as words, with no punctuation or casing.
- **Languages:** trained and tested on Hindi and English only. Whistle's other language tags are still in the vocabulary, untested.
- **Engine build:** the scores were measured with engine build 3.3.3 on x86. Other builds and other processors can move them a little.
- **Devices:** speed and memory on phones have not been measured yet.
- **Clip length:** trained on clips of up to 30 seconds. The engine accepts longer audio, but long recordings were not scored on their own.

## Responsible use

Transcripts from this model are machine-generated and can be wrong. Label them as machine-generated when you show them to people or pass them on. The Mozilla Data Collective terms for Common Voice, one of the training sources, ask that outputs of models built with that data be disclosed to downstream users as machine-generated. Have a person check a transcript before it is used in a decision about someone.

## Files

| File | What it is |
|---|---|
| `akshar-lite-v4.cact` | The model for the needle engine, 2-bit with a 4-bit encoder, 20.4 MB. The scores above were measured with this file. |
| `training/lite-v4.json` | The record of this run: settings, data by source, validation loss every 600 updates. |
| `NOTICE` | What AksharSTT Lite v4 is built from, with licences and links. |

sha256 of `akshar-lite-v4.cact`: `9a7bdf9be822d66758a13ad0ec9ef90c6ec47a88a9be1aeb526ba9d3207eff6b`

The full-precision weights it was made from are in [byteknotlabs/AksharSTT-V4](https://huggingface.co/byteknotlabs/AksharSTT-V4).

## Licence

Apache-2.0. AksharSTT v4, Whistle and Shruti are Apache-2.0 too. The training datasets are CC-BY-4.0, except MUCS 2021, which is CC-BY-SA-4.0, and Common Voice, which is CC0. `NOTICE` credits each of them.

## Citation

```bibtex
@software{aksharstt2026,
  title   = {AksharSTT: offline speech-to-text for Hindi, English and Hinglish},
  author  = {{ByteKnot Labs}},
  year    = {2026},
  version = {Lite v4},
  url     = {https://huggingface.co/byteknotlabs/AksharSTT-Lite-V4},
  note    = {Code: https://github.com/byteknotlabsai/akshar}
}
```
