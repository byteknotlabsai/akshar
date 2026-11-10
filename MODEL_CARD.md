---
license: apache-2.0
language:
- hi
- en
pipeline_tag: automatic-speech-recognition
base_model: Cactus-Compute/whistle
base_model_relation: finetune
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
model-index:
- name: AksharSTT v4
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
      value: 19.47
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
      value: 18.45
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
      value: 25.45
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
      value: 34.07
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
      value: 21.41
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
      value: 5.82
      name: WER
---

# AksharSTT v4

Offline speech-to-text for Hindi, English and Hinglish, in one 29.5 MB file made for ordinary CPUs and phones.

[Website](https://akshar.byteknotlabs.com) · [Live demo](https://akshar.byteknotlabs.com) · [GitHub](https://github.com/byteknotlabsai/akshar) · [AksharSTT Lite v4](https://huggingface.co/byteknotlabs/AksharSTT-Lite-V4) · [AksharSTT v1](https://huggingface.co/byteknotlabs/AksharSTT-V1)

AksharSTT v4 is an encoder-decoder model with 55 million parameters and 4-bit weights, run by Cactus's needle engine on the device itself. It was trained from [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle) on about 3,900 hours of speech, with [Shruti](https://huggingface.co/udgamlabs/Shruti) as its teacher. It follows [AksharSTT v1](https://huggingface.co/byteknotlabs/AksharSTT-V1); versions 2 and 3 were internal steps.

## Highlights

- Mean word error rate 20.78 over six test sets, against 21.78 for Shruti at 4 bits. It is lower than Shruti's on each of the six sets, on the same clips, scored by the same harness.
- Indian-accented English (Svarah): 21.41, against 24.68 for AksharSTT v1 and 22.45 for Shruti.
- Hinglish comes out the way people type it: Hindi in Devanagari, English words in Latin. A curated list of 2,900 English words, brands and place names decides which words the training transcripts write in Latin.
- One 29.5 MB file. Transcription runs on the device, with no network.
- On 100 clips of silence and faint noise it writes nothing.

All scores are on the full test sets with the settings given under [Accuracy](#accuracy).

## Model details

| | |
|---|---|
| Architecture | Whistle encoder-decoder: 8 encoder and 8 decoder layers of width 512, about 55 million parameters |
| Vocabulary | Whistle's, plus Shruti's 714 Devanagari pieces and the `hi` language tag |
| Weights | 4-bit, the precision it was trained at (quantisation-aware training) |
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

model = Whistle(hf_hub_download("byteknotlabs/AksharSTT-V4", "akshar-v4.cact"))
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

| Clip | AksharSTT v4 writes |
|---|---|
| MUCS 2021, a software tutorial | जैसे कि आप देख सकते हैं workspace में पांच tabs हैं |
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

Against AksharSTT v1 the mean falls by 1.50 points, most on Svarah (by 3.27) and Monsoon (by 2.31).

### Language flag

The flag matters for English speech. With `hi`, AksharSTT v4 scores 26.97 on Svarah; with `en`, 21.41 (same clips, same harness and settings otherwise). Apps should pass `en` when they know the speech is English, and `hi` for Hindi and Hinglish. Training narrows the gap: 40% of the Indian English clips were trained under the `hi` tag, and the gap fell from 9.2 points in the internal v2 to 5.6 in v4.

## Training

### Lineage

AksharSTT v4 is the fourth version of one training line that starts at Whistle. No weights come from Shruti. Shruti gave its 714 Devanagari vocabulary pieces, its transcripts of part of the training audio, and its output distributions as a distillation teacher.

| Version | Starts from | Updates | Peak learning rate | Hours of audio | GPU time |
|---|---|---|---|---|---|
| AksharSTT v1 | Whistle, vocabulary extended | 122,400 in three stages | 8e-4, 4e-4, 1e-4 | about 530 | about 11.5 h |
| v2 (internal) | AksharSTT v1 | 78,000 | 3e-4 | about 3,000 | about 6.5 h |
| v3 (internal) | v2 | 110,000 | 2e-4 | about 3,900 | about 9 h |
| AksharSTT v4 | v3's averaged weights | 64,000 | 1.5e-4 | about 3,900 | about 6 h |

Every version trained all weights with 4-bit quantisation-aware training and 8-bit activations, and with distillation from Shruti as released, weighted 0.5 against the targets. An update is about 480 seconds of padded audio. All of it ran on one RTX 5080. From v2 on, the data streamed from Google Drive while the model trained, so a server with room for about 500 hours of audio could train on 3,900.

### What changed in v4

- **Stream by share.** The trainer drew from each streamed source by a planned share, whatever the size of its shards. In v3 the large NPTEL shards had crowded out the small Hindi sources. English was half of the stream and about 45% of all batches.
- **Repaired transcripts.** An audit of about 1,000 training transcripts found faults in the converter that turns dataset transcripts into the output style. The repaired converter keeps 402,038 Vaani transcripts instead of 318,850, removes speaker labels and split numbers from NPTEL, and corrects misspelt English glosses in Rural Women.
- **English-word list.** The training transcripts write English words, brands and place names in Latin when they are on a curated list of 2,900 words (about 5,100 Devanagari spellings). A blocklist keeps Hindi words that look like English, such as पर, बस and सर, in Devanagari. The Shrutilipi transcripts were rewritten with the same list.
- **English under the Hindi tag.** 40% of the Indian English clips (Common Voice and NPTEL) trained under the `hi` tag with their Latin transcripts, against 20% in v3.
- **More Shrutilipi.** 83 hours transcribed by the internal v2 model joined the 263 hours transcribed by Shruti. A v2 transcript was kept only where it agrees with the bulletin's script within 5%.

The decisions behind these changes are D29 to D32 in [docs/decisions.md](https://github.com/byteknotlabsai/akshar/blob/main/docs/decisions.md).

### Data

| Source | Licence | Hours | Share of the stream |
|---|---|---|---|
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), English lectures | CC-BY-4.0 | 1,600 | 28% |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), Hindi | CC-BY-4.0 | 723 | 20% |
| [LibriSpeech](https://www.openslr.org/12), train-clean-360 and part of train-clean-100 | CC-BY-4.0 | 374 | 4% |
| [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part), transcribed Hindi | CC-BY-4.0 | 352 | 13% |
| [Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi), Hindi | CC-BY-4.0 | 346 | 6% |
| [Common Voice](https://datacollective.mozillafoundation.org/) Scripted Speech 26.0, South Asian English | CC0 | 158 | 18% |
| [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath), Hindi | CC-BY-4.0 | 133 | 4% |
| [MUCS 2021](https://www.openslr.org/104), Hindi-English train | CC-BY-SA-4.0 | 92 | on disk |
| [Rural Women ASR v2](https://huggingface.co/datasets/ai4bharat/Rural_Women_ASR_v2), Hindi | CC-BY-4.0 | 76 | 5% |
| [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL), Hindi lectures | CC-BY-4.0 | 60 | on disk |
| [Common Voice](https://datacollective.mozillafoundation.org/) 27.0, Hindi | CC0 | 9 | 2% |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), Hindi train | CC-BY-4.0 | 3 | on disk |
| Silence and noise, generated by akshar's code | | 2 | on disk |

- **Hours** count each clip of v4's run once: about 3,930 in all.
- **Batches:** 70% came from the stream at the shares shown. The other 30% came from 230 hours kept on disk: MUCS, NPTEL's Hindi lectures and 53 hours of its English ones, 22 hours of LibriSpeech train-clean-100, FLEURS and the silence clips.
- **Targets:** each dataset's own transcripts where they are verbatim, converted to the output style with numbers spelled out. Shruti's transcripts for MUCS, the NPTEL clips on disk and 263 hours of Shrutilipi; the internal v2's for the other 83 hours of Shrutilipi.
- **Test clips removed:** every training clip that matches a test clip by its audio, or by its text when the text has three or more words. Clips over 30 seconds were dropped.

### The v4 run

- **Start:** v3's averaged weights. 64,000 updates at a learning rate of 1.5e-4 with cosine decay.
- **Loss:** cross-entropy on the targets plus distillation from Shruti (4-bit, as released) at weight 0.5, temperature 1.0.
- **Validation cross-entropy** on 2,691 held-out clips: 0.2112 at the start, 0.2191 at update 20,000, 0.2111 at the end.
- **Audio:** about 5,900 hours passed through the run, 4,050 of them from the stream.
- **Compute:** two night windows on one RTX 5080, about 6 GPU-hours. Peak GPU memory 4.9 GB.
- **Weight average:** the run also kept a moving average of the weights from update 38,400. It scored the same mean on the test sets. This release is the final weights.

The full record of the run is `training/v4.json`.

## Intended use

- Transcribing Hindi, English and Hinglish speech on phones, laptops and servers without a network: dictation, voice notes, captions, voice input and search over recordings.
- Apps that must keep audio on the device.
- A starting point for further training with the [akshar](https://github.com/byteknotlabsai/akshar) code, from the full-precision weights.

Out of scope: languages other than Hindi and English, punctuated or formatted text, translation, and telling speakers apart.

## Limitations

- **Hinglish tutorials (MUCS):** 34 to 35 WER for every model in the table. MUCS references are the narrators' scripts rather than verbatim transcripts, and they write the same English word in either script, so part of that error is in the references.
- **Accented Hindi (Lahaja):** 25.45, the set where v4 lost the most against the internal v3 it was trained from (24.87; FLEURS lost 0.18). v4 writes about 10% more Latin words on Lahaja than v3 did, mostly wrong ones. The larger English share in training, part of it under the `hi` tag, is the likely cause.
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
| `akshar-v4.cact` | The model for the needle engine, 4-bit, 29.5 MB. The scores above were measured with this file. |
| `akshar-v4.safetensors` | The full-precision weights in akshar's checkpoint format (config and vocabulary in its metadata), 222 MB, for further training with the akshar code. |
| `training/v4.json` | The record of the v4 run: settings, data by source, validation loss every 1,000 updates. |
| `NOTICE` | What AksharSTT v4 is built from, with licences and links. |

sha256 of `akshar-v4.cact`: `076782372806bc2f6a74299d857fd37b22ec1609ddfbab5a02c2078d5493bd75`

## Licence

Apache-2.0. Whistle and Shruti are Apache-2.0 too. The training datasets are CC-BY-4.0, except MUCS 2021, which is CC-BY-SA-4.0, and Common Voice, which is CC0. `NOTICE` credits each of them.

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
