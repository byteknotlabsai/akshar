# akshar

An open-source, offline speech-to-text model for Hindi, English and Hinglish, small enough for phones and ordinary CPUs.

**Status (2026-10-31):** Phases 0 and 1 are done. The evaluation harness reproduces Shruti's published numbers to within 0.22 WER on all seven test sets ([docs/eval.md](docs/eval.md)). We can train all of Whistle's weights and export 4-bit and 2-bit models ([docs/toolchain.md](docs/toolchain.md)). No akshar model is trained yet. Any number in this repo is someone else's measurement unless a table says it is ours.

## Goal

Beat [Shruti](https://huggingface.co/udgamlabs/Shruti)'s accuracy with a commercially clean model, released in two sizes so each app can choose:

| | Shruti today | akshar (4-bit) | akshar-lite (2-bit) |
|---|---|---|---|
| Model file | 29.5 MB | ≤ 30 MB, hard cap 40 MB | about 17 MB |
| Accuracy | table below | no test set more than 0.5 WER worse than Shruti, and a lower average WER | published for every test set; provisional bar: average within 2 WER of akshar |
| Invented words on 100 silent clips | 0 | 0 | 0 |
| Peak RAM on an Android phone | 80 MB (measured) | ≤ 90 MB | about 65–70 MB (estimate) |
| Runs on | Needle engine: Windows (x64, ARM64), macOS (Apple silicon), Linux, Android, iOS, WebAssembly | same | same |

Training needs Linux, but running the model does not: Windows and Mac apps embed the engine directly (decision D12).

akshar carries the "no compromise on output" promise. akshar-lite trades some accuracy for size, if 2 bits turns out to cost any; if it costs nothing, akshar-lite becomes the default. Peak RAM is roughly the file size plus about 50 MB of working memory.

Shruti's published word error rates (%), the bar we have to clear:

| Monsoon Hindi | FLEURS Hindi | Lahaja (accented Hindi) | MUCS Hinglish | Svarah (Indian English) | LibriSpeech clean |
|---|---|---|---|---|---|
| 20.46 | 19.92 | 26.28 | 35.21 | 22.47 | 5.95 |

We score both models with the same harness and protocol ([roadmap](docs/roadmap.md#evaluation-protocol)) before claiming anything.

## Plan in brief

1. **Eval harness first.** Our harness must reproduce Shruti's numbers before we trust it with ours.
2. **Track A:** compress Shruti (Apache-2.0) to 2 bits, using quantisation-aware training with distillation from its full-precision weights. This develops the 2-bit recipe and measures what 2 bits costs, early and cheaply. The result is research-only, because some of Shruti's training data fails our commercial licence check ([data/LICENSES.md](data/LICENSES.md)).
3. **Track B:** train akshar from Cactus Whistle on commercially clean data only, at 4 bits. Then make akshar-lite from it with Track A's 2-bit recipe. Shruti is the benchmark. These are the models we release.
4. **Output styles:** the text layer that renders `roman` and `devanagari` is built in parallel with training. Then come **phones** and **release**.

Details and exit checks: [docs/roadmap.md](docs/roadmap.md).

## Output styles

akshar gives the same transcript in three styles. An app asks for one, or takes all three at once:

| Style | Example 1 | Example 2 |
|---|---|---|
| `roman` | tum kaise ho | tumhara laptop accha lag raha hai |
| `devanagari` | तुम कैसे हो | तुम्हारा लैपटॉप अच्छा लग रहा है |
| `mixed` | तुम कैसे हो | तुम्हारा laptop अच्छा लग रहा है |

The model itself always writes `mixed`: Hindi in Devanagari, English words in Latin, numbers as words, no punctuation. That is Shruti's convention, so scores compare directly.

`mixed` is the only style that records which words were spoken in English. The other two can be built from it, but not the other way round, so a small text layer shipped with the model renders `roman` and `devanagari`. Capitals (`Laptop`), punctuation and digits are options on top of any style. Reasons: decisions D3 and D11.

## Repository map

| Path | What it is |
|---|---|
| [docs/research/2026-10-31-foundations.md](docs/research/2026-10-31-foundations.md) | Verified facts about Whistle, Needle and Shruti, with sources |
| [docs/roadmap.md](docs/roadmap.md) | Phases, exit checks, evaluation protocol, compute plan, risks |
| [docs/eval.md](docs/eval.md) | How to run the evaluation harness, the protocol's details, and Shruti's scores through it |
| [akshar/eval/](akshar/eval/) | The evaluation harness: test-set download, decoding in the needle engine, scoring |
| [docs/toolchain.md](docs/toolchain.md) | How training and export work, and what Phase 1 measured |
| [akshar/model/](akshar/model/), [akshar/train/](akshar/train/), [akshar/data/](akshar/data/) | Checkpoints, quantisation and `.cact` export; full-weight training; training manifests |
| [docs/decisions.md](docs/decisions.md) | Decision log |
| [docs/setup-ubuntu.md](docs/setup-ubuntu.md) | First-time setup on Ubuntu (where all building happens) |
| [data/LICENSES.md](data/LICENSES.md) | Dataset licence ledger. No dataset is used for training without an entry. |

## License

akshar is open source under the [Apache License 2.0](LICENSE). We'll release the model weights under Apache-2.0 too, with every training dataset credited in [NOTICE](NOTICE) as its licence requires.

## Credits

akshar builds on [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle) (Apache-2.0, Cactus Compute) and [Shruti](https://huggingface.co/udgamlabs/Shruti) (Apache-2.0, Udgam Labs). See [NOTICE](NOTICE).
