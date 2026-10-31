# Setting up akshar on Ubuntu 24.04

Ubuntu is for training and evaluation only. The finished model runs on Windows, Mac and phones without it (decision D12).

Do this once on the laptop's Ubuntu install, and again on the 32 GB machine. All commands come from each tool's own documentation. They have not been run on this project yet, so stop and note anything that fails.

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

## 5. Needle with training and GPU support

```bash
uv pip install "cactus-needle[train,gpu]" huggingface_hub jiwer soundfile
```

## 6. Turn telemetry off (decision D6)

`cactus-needle` reports anonymous usage by default.

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

## 11. Start Phase 0

Open Claude Code in `~/akshar`. `CLAUDE.md` carries the project context. Ask it to start Phase 0 (the eval harness) from [roadmap.md](roadmap.md).

## On the 32 GB machine

Repeat steps 1 to 7. A newer RTX card may need a newer driver than the laptop's, so trust `jax.devices()` rather than assuming.
