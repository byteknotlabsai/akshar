# akshar: notes for Claude

akshar is an offline Hindi/English/Hinglish speech-to-text model for CPUs and phones. The goal is to beat Shruti's accuracy with a model cleared for commercial use, in two variants: akshar (4-bit, file ≤ 30 MB, hard cap 40 MB) and akshar-lite (2-bit with a 4-bit encoder, about 20 MB, accuracy published per test set; D24). Before proposing work, read `README.md`, `docs/decisions.md` and the current phase in `docs/roadmap.md`. Verified facts and sources are in `docs/research/`.

## Ground rules

- **Where work happens:**
  - Docs may be edited on Windows.
  - Building, training and evaluation happen on Ubuntu, because JAX's CUDA builds are Linux-only.
  - Running the finished model needs no Linux: it ships in Windows and Mac (Apple silicon) apps and on phones through the Needle engine (D12).
  - Keep the text layer portable; don't make it Python-only.
- **Compute:** free only.
  - Laptop RTX 3050 Laptop 4 GB.
  - Kaggle and Colab free tiers.
  - An RTX 5080 (16 GB) on the owner's Ubuntu server, evenings only, after its llama.cpp service is stopped; reached over SSH via Tailscale, working in `~/user1` (D20). Never start on its GPU while another process uses it. A window that stops coot-moe must end before 03:00, when the owner's cleanup deletes stopped containers (D23).
  - Never plan around paid GPUs.
- **Telemetry:** export `NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1` before any needle command, because the package reports usage by default.
- **Licences:**
  - No dataset enters training without a row in `data/LICENSES.md` marked `admit`, `admit (D21)`, `admit (D26)`, `admit, caveat` or `hold, train (D19)`.
  - A model trained on a `hold, train` source must not be released until a lawyer has cleared that source (D19). MUCS 2021 and Common Voice, the last two, were cleared by the owner's legal guidance (D26). Keep every clip tagged with its source so one can be dropped.
  - Track A weights (built from Shruti) are research-only (D9). Never present them as releasable.
- **Contamination:** remove every eval clip from training data, including data added later: by audio hash always, and by normalised-text hash when the text has three or more words (D17).
- **Output convention:**
  - The core model writes the `mixed` style: Hindi in Devanagari, English words in lowercase Latin, numbers as words, no punctuation.
  - The product offers three styles (`roman`, `devanagari`, `mixed`; D11), all rendered by the text layer from that one output.
  - Other scripts and formatting never go into the acoustic model's training targets.
- **Reporting:**
  - Scores come from the harness, `python -m akshar.eval` ([docs/eval.md](docs/eval.md)). D13 fixes its normaliser, scorers and pins.
  - Never quote a WER without its test set, normaliser and decoding settings (beam size, language flag).
  - Never claim akshar beats or matches Shruti unless our harness scored both on the same clips.
- **Big files:** datasets live outside the repo (e.g. `~/akshar-data`). `models/`, `*.cact` and `*.safetensors` are never committed. Google Drive holds what we create (pseudo-labels, packed shards, checkpoints), uploaded as large files; training reads only from local disk (D14).
- **Decisions:** record a new decision as a new entry at the bottom of `docs/decisions.md`; don't rewrite old entries.
