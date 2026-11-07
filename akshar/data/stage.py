"""Build a training source shard by shard and keep it on Google Drive, so the GPU server never holds all of it (D14).

    python -m akshar.data.stage indicvoices_hi --remote gdrive:akshar/p3 --rclone-config ~/user1/rclone/rclone.conf

It runs akshar.data.build (download, keep 0.5 to 30 s clips, blocklist check, 16 kHz FLAC) and, beside it, packs each
finished shard's clips into one uncompressed tar, uploads it with rclone, and deletes the local copy. Drive handles a
few large files far better than many small ones. Lossless: the tar holds the same FLAC bytes. Resumable: the build
skips finished shards, and shards listed in staged.txt are not uploaded again.

On the remote, under <remote>/<source>/:
    clips/<shard>.tar   that shard's clips, as clips/<shard>/<id>.flac
    manifest.jsonl      one clip per line, as akshar.data.build writes it
    progress.json, blocked.jsonl, summary.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from ..eval.paths import DATA_ROOT
from . import build as builder


class Stager:
    def __init__(self, source: str, out: Path, remote: str, rclone: list[str]):
        self.source, self.out, self.remote, self.rclone = source, out, f"{remote.rstrip('/')}/{source}", rclone
        self.ledger = out / "staged.txt"
        self.staged = set(self.ledger.read_text().split()) if self.ledger.exists() else set()
        self.stop = threading.Event()
        self.error: Exception | None = None  # the last staging failure, if any

    def _run(self, *args: str, attempts: int = 6) -> None:
        for attempt in range(attempts):
            if subprocess.run([*self.rclone, *args], stdout=subprocess.DEVNULL).returncode == 0:
                return
            time.sleep(30 * (attempt + 1))  # a dropped link usually comes back
        raise RuntimeError(f"rclone {' '.join(args[:2])} failed {attempts} times")

    def _done_shards(self) -> list[str]:
        path = self.out / "progress.json"
        try:
            return [Path(s).stem for s in json.loads(path.read_text())["done"]] if path.exists() else []
        except json.JSONDecodeError:  # caught mid-rewrite by the builder; read it next round
            return []

    def stage_ready(self) -> int:
        n = 0
        for stem in self._done_shards():
            folder = self.out / "clips" / stem
            if stem in self.staged:
                shutil.rmtree(folder, ignore_errors=True)
                continue
            if not folder.exists():
                raise SystemExit(f"{folder} is gone but was never staged")
            tar = self.out / "staging" / f"{stem}.tar"
            tar.parent.mkdir(exist_ok=True)
            with tarfile.open(tar, "w") as handle:
                handle.add(folder, arcname=f"clips/{stem}")
            self._run("copyto", str(tar), f"{self.remote}/clips/{stem}.tar")
            with open(self.ledger, "a") as handle:
                handle.write(stem + "\n")
            self.staged.add(stem)
            shutil.rmtree(folder)
            tar.unlink()
            n += 1
        return n

    def upload_lists(self) -> None:
        # The builder appends to these while it runs, and rclone refuses a file that grows mid-upload: send a snapshot.
        snap = self.out / "staging" / "lists"
        snap.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.jsonl", "progress.json", "blocked.jsonl"):
            if (self.out / name).exists():
                shutil.copyfile(self.out / name, snap / name)
                self._run("copyto", str(snap / name), f"{self.remote}/{name}")

    def loop(self) -> None:
        while not self.stop.is_set():
            try:
                if self.stage_ready():
                    self.upload_lists()
            except Exception as exc:  # keep going; the next round retries, and the main thread reports the last error
                self.error = exc
                print(f"staging: {exc}", flush=True)
            self.stop.wait(20)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.data.stage", description=__doc__.split("\n\n")[0])
    parser.add_argument("source")
    parser.add_argument("--remote", required=True, help="rclone path that holds one folder per source")
    parser.add_argument("--hours", type=float, default=1e9, help="stop after the shard that reaches this (default: all)")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--rclone", default="rclone")
    parser.add_argument("--rclone-config")
    parser.add_argument("--bwlimit", default="15M", help="upload speed cap, so the server's other services keep bandwidth")
    args = parser.parse_args(argv)
    rclone = [args.rclone, "--bwlimit", args.bwlimit, "--drive-chunk-size", "128M"]
    if args.rclone_config:
        rclone += ["--config", str(Path(args.rclone_config).expanduser())]
    out = DATA_ROOT / "train" / args.source
    out.mkdir(parents=True, exist_ok=True)
    stager = Stager(args.source, out, args.remote, rclone)
    stager.stage_ready()  # whatever a previous run finished
    worker = threading.Thread(target=stager.loop, daemon=True)
    worker.start()
    started = time.time()
    summary = builder.build(args.source, args.hours, out, args.fraction)
    stager.stop.set()
    worker.join()
    stager.stage_ready()
    stager.upload_lists()
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    stager._run("copyto", str(out / "summary.json"), f"{stager.remote}/summary.json")
    print(json.dumps({**summary, "staged_shards": len(stager.staged), "minutes": round((time.time() - started) / 60, 1)}))


if __name__ == "__main__":
    main()
