# Setting up akshar on Ubuntu

Ubuntu is for training and evaluation only. The finished model runs on Windows, Mac and phones without it (decision D12).

Do this once on the laptop's Ubuntu install, and again on the 32 GB machine. The laptop (Ubuntu 26.04) was set up from this guide on 2026-10-31, except the microphone test in step 9. The harness has not needed step 1's ffmpeg or git-lfs so far. On a new machine, stop and note anything that fails.

## 1. System packages

```bash
sudo apt update
```

```bash
sudo apt install -y git git-lfs build-essential curl ffmpeg libsndfile1
```

## 2. Get the repo

```bash
git clone https://github.com/ByteKnotLabs/akshar.git ~/akshar
```

## 3. NVIDIA driver

```bash
nvidia-smi
```

It should list "NVIDIA GeForce RTX 3050 Laptop GPU" with 4096 MiB. If the command fails, install the recommended driver, reboot and check again:

```bash
sudo ubuntu-drivers install
```

## 4. uv and Python 3.12

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, then create the environment:

```bash
cd ~/akshar && uv venv --python 3.12 && source .venv/bin/activate
```

## 5. Install the pinned dependencies

```bash
uv sync --extra gpu
```

This installs exactly what `uv.lock` records: `cactus-needle` 3.2.8 with training and GPU support, the scorers, and the `akshar` package itself. The needle version is pinned because it decides the engine build (3.2.8 loads engine 3.3.3), and engine builds move WER slightly. Tested 2026-10-31 on Ubuntu 26.04 with an RTX 3050 Laptop GPU (driver 595, CUDA 13.2); JAX 0.12 found the GPU with no extra steps.

## 6. Turn telemetry off (decision D6)

`cactus-needle` reports anonymous usage by default. The eval harness turns it off for its own runs, but set it for your shell too:

```bash
echo 'export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1' >> ~/.bashrc && source ~/.bashrc
```

## 7. Check that JAX sees the GPU

```bash
python -c "import jax; print(jax.devices())"
```

You should see a CUDA device. If it shows only `CpuDevice`, install the CUDA build of JAX from the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html), then run the check again.

## 8. Download the reference model

```bash
hf download udgamlabs/Shruti shruti.cact shruti.safetensors --local-dir models/shruti
```

This is 29.5 MB plus 222 MB. `models/` is git-ignored.

## 9. Smoke test

Record five seconds of yourself speaking Hindi or Hinglish at 16 kHz mono, then transcribe it:

```bash
arecord -f S16_LE -r 16000 -c 1 -d 5 test.wav
```

```bash
python -c "from needle import Whistle; print(Whistle('models/shruti/shruti.cact').transcribe('test.wav'))"
```

## 10. Disk space

Keep datasets outside the repo, for example in `~/akshar-data`. Check free space first:

```bash
df -h ~
```

Track A needs tens of GB and Track B needs hundreds. 2,732 h of 16 kHz audio is about 315 GB as raw PCM and about half that as FLAC.

## 11. Run the eval harness

Phase 0's harness is in `akshar/eval/`. [eval.md](eval.md) explains how to run it and what it measured.

Two test sets, Lahaja and Svarah, are gated on Hugging Face. On each dataset page (ai4bharat/Lahaja and ai4bharat/Svarah), fill in and submit the access form while signed in. Approval is automatic, and the page then confirms you have access. Then log in here with a token from the same account. Use a classic "Read" token. A fine-grained token works only with "Read access to contents of all public gated repos you can access" ticked; without it, downloads fail with "not in the authorized list".

```bash
hf auth login
```

## On the 32 GB machine

Repeat steps 1 to 7. A newer RTX card may need a newer driver than the laptop's, so trust `jax.devices()` rather than assuming.

This machine runs Phase 3 (D14). Before using it, record its specs in D14's "Open" line:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv && free -h && nproc && df -h ~
```

Download training data on this machine directly from Hugging Face. Copying it from the laptop or from Google Drive over the internet link is far slower.
