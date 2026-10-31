# Dataset licence ledger

Rule (decision D5): **no dataset is used for training unless it is listed here with status `admit` or `admit, caveat`.** Test sets are `eval only` and must also be blocked from training by text and audio hash.

Checked 2026-10-31 from web pages only (dataset cards, licence pages, papers). This is a research record, not legal advice. Before any weights are released, a person with legal training should review every `caveat` and `hold` line.

## Summary

- **Four of Shruti's ten training sources are on hold:**
  - Shrutilipi: All India Radio copyright.
  - AI4Bharat NPTEL: NPTEL's own licence may be non-commercial.
  - MUCS 2021: share-alike.
  - Common Voice: new Mozilla Data Collective terms.
  - **So Track A models, which start from Shruti's weights, are research-only until these are resolved (decision D9).** Track B is the path to a commercially clean akshar.
- **Clean, transcribed Hindi pool:** about 2,000 h before IndicVoices and Rural Women (hours not published).
  - Vaani: 1,004 h.
  - SPRING-INX: ~351 h.
  - Kathbath: ~150 h.
  - FLEURS: ~10 h.
  - IndicVoices.
  - Rural Women ASR v2.
- **Pseudo-labelling pool:** about 14,900 h of untranscribed Vaani Hindi audio (CC-BY-4.0). This is the biggest single lever for Track B.
- **English:** plenty of clean data (LibriSpeech, MLS English, People's Speech CC-BY configs), but almost none of it is Indian-accented. **Clean Indian English is the main data gap**, because NPTEL and Common Voice's Indian-accent clips are both on hold.
- **Natural Hinglish:** no large permissive corpus exists apart from MUCS (share-alike, on hold). Code-mixed speech exists inside IndicVoices, Vaani and SPRING-INX (which targeted about 10% code-mixing). We will have to mine it from those.

## Ledger

Hours are as published; n/s means not stated.

| Dataset | Licence (as verified) | Status | Hours (hi / en / mixed) | Domain | Access | In Shruti? |
|---|---|---|---|---|---|---|
| IndicVoices (hi) | CC-BY-4.0; the paper allows commercial use | admit | hi n/s | extempore, conversational, read | HF gate (auto) | yes |
| Vaani (hi) | CC-BY-4.0; the gate form lists commercial use | admit | hi 1,004 transcribed, ~14,924 audio | image-prompted spontaneous | HF gate (form) | yes |
| SPRING-INX Hindi | CC-BY-4.0 | admit | hi ~351 | monologue, mock conversation | AIKosh login | no |
| LibriSpeech | CC-BY-4.0 | admit | en ~1,000 | audiobooks | open | yes |
| MLS English | CC-BY-4.0 | admit | en 44,660 | audiobooks | open | no |
| Kathbath (hi) | tag CC-BY-4.0; card body says CC0 packaging | admit, caveat | hi ~150 | read, phone | HF gate (auto) | yes |
| Rural Women ASR v2 (hi) | README says CC-BY-4.0; no licence tag | admit, caveat | hi n/s (30.8k utterances) | rural extempore | open | yes |
| FLEURS (hi train) | CC-BY-4.0; sentences from CC-BY-SA-4.0 FLoRes | admit, caveat | hi ~10 | read | open | yes |
| People's Speech (CC-BY configs only) | CC-BY | admit, caveat | en 30k+, mostly US | mixed archive.org | open | no |
| IndicVoices-R | CC-BY-4.0 | admit, low value | overlaps IndicVoices | TTS-enhanced extempore | HF gate (auto) | no |
| Shrutilipi (hi) | CC-BY-4.0 label, but audio is All India Radio | **hold** | hi 1,620 | news broadcast | HF gate (auto) | yes |
| AI4Bharat NPTEL / NPTEL2020 | CC-BY-4.0 label; NPTEL shows BY-SA or BY-NC-SA | **hold** | en up to 15,700; hi audio 1,434 | lectures | HF / GitHub | yes |
| MUCS 2021 Hindi-English (OpenSLR 104) | CC-BY-SA-4.0; the HF mirror mislabels it BY | **hold** | mixed ~90 train | tech tutorials | open | yes |
| Common Voice 17 / 27 (hi, en) | CC0 plus Mozilla Data Collective terms (updated 2026-10-28) | **hold** | hi 15.7; en 2,827 (5.9% Indian accent) | read | MDC account | yes (en) |
| MUCS Hindi (OpenSLR 103) | licence link dead (Microsoft) | **hold** | hi ~95 | read, 8 kHz | open | no |
| Bhashini / ULCA | set per dataset (NC and ND allowed) | **hold** | varies | varies | login | no |
| Gramvaani (OpenSLR 118) | free for academic use only | exclude | hi 108 labelled, ~1,000 unlabelled | telephone | open | no |
| GigaSpeech | non-commercial terms of access | exclude | en 10,000 | podcasts, YouTube | gated | no |
| CS-FLEURS | CC-BY-NC-4.0 | exclude | mostly synthetic | TTS | open | no |
| sonexis Hinglish conversations | CC-BY-NC-4.0 | exclude | < 1k clips | conversation | open | no |
| YODAS | CC-BY-3.0 as declared by YouTube uploaders | exclude | en ~169,900; hi ~484 | YouTube | open | no |
| Monsoon hi-IN (public split) | CC-BY-4.0 | eval only | hi 1.33 (753 clips) | phone conversations | open | eval |
| FLEURS hi test | CC-BY-4.0 | eval only | 418 clips | read | open | eval |
| Lahaja | tag MIT; body CC-BY-4.0 | eval only | hi 12.5 | read + extempore | HF gate (auto) | eval |
| Svarah | CC-BY-4.0 | eval only | en 9.6 | read + spontaneous | HF gate (auto) | eval |
| MUCS 2021 Hindi-English test | CC-BY-SA-4.0 | eval only | ~5 | tech tutorials | open | eval |
| Vistaar benchmark sets | no data licence stated | eval only | n/a | benchmarks | GitHub | n/a |

**Status meanings:**
- `admit`: verified permissive licence, no known conflict.
- `admit, caveat`: permissive, with a documented caveat for legal review before release.
- `hold`: commercial use unclear; not used for Track B training until resolved.
- `exclude`: non-commercial or no permission.
- `eval only`: a test set, never trained on.

## Notes on held and caveated sources

- **Shrutilipi.** Mined from All India Radio news bulletins. AIR's [copyright policy](https://newsonair.gov.in/copyright-policy/) allows only "own personal, non commercial use"; any other use needs prior written permission from AIR's News Services Division (checked directly 2026-10-31). Transcripts are bulletin scripts read off PDFs by OCR, not verbatim. This is the largest Hindi source Shruti used.
- **NPTEL.** AI4Bharat's [NPTEL set](https://huggingface.co/datasets/ai4bharat/NPTEL) is labelled CC-BY-4.0, but [nptel.ac.in](https://nptel.ac.in/) shows a CC BY-NC-SA badge. The [Creative Commons wiki](https://wiki.creativecommons.org/wiki/OER_Case_Studies) says NPTEL dropped NC in 2014, which would make it BY-SA. Either way the CC-BY relabel is doubtful.
- **MUCS 2021 Hindi-English.** CC-BY-SA-4.0 on [OpenSLR 104](https://www.openslr.org/104/). The [HF mirror](https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish) Shruti used says CC-BY-4.0. The data has quality problems of its own: transcripts are narrators' scripts, the same English word appears in both scripts, timestamps are misaligned, and there are no speaker IDs.
- **Common Voice.** Data is CC0, but downloads now go through Mozilla Data Collective, whose terms (updated 2026-10-28) ban re-hosting and re-identification. They also ask that outputs of models built with the data contain no personal data and are "clearly disclosed to downstream users as machine-generated". How that applies to an Apache-2.0 model needs a reading of the full terms.
- **Kathbath.** The licence tag says CC-BY-4.0, while the [card](https://huggingface.co/datasets/ai4bharat/kathbath) and [IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB) license the packaging as CC0. The read sentences came from a web crawl (IndicCorp) that AI4Bharat doesn't own.
- **Rural Women ASR v2.** Only the README states CC-BY-4.0, and there's no licence tag. Its sibling Bhojpuri set is CC-BY-SA-4.0. English loanwords are bracketed (e.g. `[mobile]`).
- **FLEURS.** Audio and card are CC-BY-4.0, but the sentences are translations from [FLoRes](https://huggingface.co/datasets/facebook/flores), which is CC-BY-SA-4.0.
- **People's Speech.** Use only the CC-BY configs. Some transcripts are probably machine-generated, and part of the data is poorly aligned.
- **Share-alike, open question.** Does a CC-BY-SA licence on training data reach the trained weights? This affects MUCS, FLoRes text inside FLEURS, and possibly NPTEL.

## Notes on admitted sources

- **IndicVoices** ([card](https://huggingface.co/datasets/ai4bharat/IndicVoices), [paper](https://arxiv.org/html/2403.01926)).
  - Recorded at 44.1 kHz; role-play conversations went over an 8 kHz phone bridge.
  - Transcripts come in verbatim, normalised and unsanitised forms with 89 annotation tags.
  - Shruti converted its bracketed English words to Latin script. Check how they're written in the raw data.
- **Vaani** ([card](https://huggingface.co/datasets/ARTPARK-IISc/Vaani), [site](https://vaani.iisc.ac.in/)).
  - 16 kHz phone-app audio; numbers are written as words.
  - Tags include `[unintelligible]`, `<PAUSE>` and `misspelling{correction}`.
  - The gate form asks your purpose, and "Commercial" is one of the options.
- **SPRING-INX Hindi** ([AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/spring_inx_hindi.html)).
  - Not on Hugging Face.
  - 316/30/5 h train/dev/test.
  - Phone audio upsampled from 8 kHz; English spellings were corrected.
- **LibriSpeech** ([OpenSLR 12](https://www.openslr.org/12/)) and **MLS English** ([OpenSLR 94](https://www.openslr.org/94/)): CC-BY-4.0, 16 kHz audiobooks.

## Monsoon (the Open ASR leaderboard's Hindi test set)

- **Made by:** Voice Arena. Added to the Hugging Face Open ASR leaderboard on 2026-08-28.
- **Audio:** unscripted two-person phone conversations recorded on 315 handset models.
- **Public split:** [hi-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN), CC-BY-4.0, 753 clips (1.33 h). This is the set Shruti reports. A private split is scored only through a pull request to the [leaderboard repo](https://github.com/huggingface/open_asr_leaderboard).
- **References are lattices:** each word group lists every accepted spelling (e.g. `इम्प्रूव | improve`, `एक | 1`), so our mixed-script output is not penalised for writing English words in Latin.
- **Scoring:** the [voi-oiwer](https://pypi.org/project/voi-oiwer/) package (MIT). Report corpus-level WER (all errors ÷ all reference words), not an average of per-clip scores.

## Teacher models (pseudo-labels)

A teacher's licence covers its outputs, but some teachers were trained on sources this ledger holds or excludes. Whether that carries into pseudo-labels is an open legal question.

| Teacher | Weights licence | Trained on held or excluded sources? |
|---|---|---|
| [Koyal-hi-120M](https://huggingface.co/adalat-ai/koyal-hi-120m-1.0) | CC-BY-4.0, gated | Yes: Shrutilipi (hold), Common Voice (hold), OpenSLR 118 Gramvaani (exclude) |
| [IndicConformer-600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | MIT, gated | Unknown; the card lists no training data |
| Whisper large-v3 | MIT | Unknown; training data undisclosed |
| [Shruti](https://huggingface.co/udgamlabs/Shruti) | Apache-2.0 | Yes: Shrutilipi, NPTEL, MUCS, Common Voice (all hold) |

Rule: record which teacher labelled every pseudo-labelled clip, so that one teacher's labels can be dropped and the model retrained if legal review requires it.

## Text resources (output styles)

These are used by the text layer that renders `roman` and `devanagari` (D11). Anything shipped inside an app must have its licence honoured there too.

| Resource | What it gives us | Licence | Status |
|---|---|---|---|
| [IndicXlit](https://github.com/AI4Bharat/IndicXlit) | word-level transliteration models, Roman→Hindi and Hindi→Roman (~11M parameters) | MIT (code and models) | admit |
| [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) | ~1.3M Hindi–Roman word pairs | CC-BY (manually collected) and CC0 (mined); includes pairs taken from Dakshina | admit, caveat |
| [Dakshina](https://github.com/google-research-datasets/dakshina) | Hindi romanisation lexicon (25k words, each spelling counted by how many native speakers chose it) and romanised sentences | CC-BY-SA 4.0 | admit, caveat |

Notes:
- A lexicon built from Dakshina, or from Aksharantar's Dakshina-derived pairs, is share-alike. If we ship it, that file goes out under CC-BY-SA with attribution. Keep it as a separate file from the model and the code.
- IndicXlit reports 52–60% top-1 accuracy for Hindi in both directions. That is measured against a single reference spelling, so it understates usable output. Trust our own per-style WER, not this number.

## Adding a dataset

1. Read the licence on the primary source (dataset card or licence page, plus the paper or website), not a mirror.
2. Add a row above, with links in the notes.
3. If anything conflicts (tag vs card body, mirror vs original, underlying content owner), mark it `hold`.
4. Record new decisions in `docs/decisions.md`.
