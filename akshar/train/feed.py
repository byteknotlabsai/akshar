"""Stream training shards from Google Drive through a small folder while a run trains (akshar.train.distill --pool),
so that one run can learn from far more audio than the disk holds (D14 keeps the data on Drive).

    python -m akshar.train.feed CONFIG.json

CONFIG names the pool folder, its size on disk, the free space to leave, rclone, and the sources:

    {"pool": "/work/pool", "pool_gb": 12, "floor_gb": 10.5, "labels": "mix", "seed": 0,
     "rclone": ["rclone", "--config", "/work/rclone.conf"],
     "sources": [{"name": "indicvoices_hi", "remote": "gdrive:akshar/p3/indicvoices_hi/clips",
                  "manifest": "/work/data3/train/indicvoices_hi/manifest.jsonl",
                  "labels": "/work/data3/train/indicvoices_hi/labels-mix.jsonl", "share": 0.35,
                  "exclude": ["/work/data/train/nptel_en/manifest.jsonl"]}]}

A source's shards are its Drive tars (as akshar.data.stage writes them: clips/<shard>/<id>.flac) that hold at least
one labelled clip of 30 s or less whose id no `exclude` manifest lists. The next shard comes from the source furthest
below its share of the pool's hours, and within a source from the shards fetched least often, so every shard is seen
once before any is seen twice. A shard is unpacked into <pool>/<source>__<shard>/ beside its own manifest.jsonl and
labels-<labels>.jsonl (unlabelled clips are deleted), then added to <pool>/pool.json.

When the pool is full, its oldest shard leaves pool.json and is deleted only once the run has written an inuse.json,
after that, that no longer lists it, so the run never loses a file it may still read. The config is re-read and
Drive re-listed every few minutes: sources, shares and shards still being staged can join during a run. Every shard
fetched is logged in <pool>/fetched.jsonl. Restartable: shards already in the pool stay.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

GRACE_S = 30  # after the run lets go of a shard, time for its loader threads to finish the batch they were building
RELIST_S = 300


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def rows_of(path) -> list[tuple[str, dict]]:
    """(line, row) for each complete line of a JSONL file that may still be growing: a half-written line is skipped."""
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            try:
                out.append((line, json.loads(line)))
            except ValueError:
                continue
    return out


def folder_bytes(folder: Path) -> int:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


class Source:
    def __init__(self, spec: dict, rclone: list[str]):
        self.rclone = rclone
        self.by_shard: dict[str, list[tuple[str, str, float]]] = {}
        self.sizes: dict[str, int] = {}
        self.visits: dict[str, int] = {}
        self.signature = None
        self.update(spec)

    def update(self, spec: dict) -> None:
        self.spec, self.name, self.share = spec, spec["name"], float(spec["share"])
        self.remote = spec["remote"].rstrip("/")

    def load(self) -> None:
        """(Re)read the manifest and labels when either has changed."""
        paths = [Path(self.spec["manifest"]), Path(self.spec["labels"]), *map(Path, self.spec.get("exclude", []))]
        signature = [(str(p), p.stat().st_mtime) for p in paths]
        if signature == self.signature:
            return
        labels = {row["id"]: line for line, row in rows_of(self.spec["labels"])}
        excluded = {row["id"] for path in self.spec.get("exclude", []) for _, row in rows_of(path)}
        by_shard: dict[str, list] = {}
        for line, row in rows_of(self.spec["manifest"]):
            label = labels.get(row["id"])
            if label is None or row["id"] in excluded or row["duration"] > 30.0:
                continue
            shard = row["audio"].split("/")[1]
            by_shard.setdefault(shard, []).append((line, label, row["duration"]))
        self.by_shard, self.signature = by_shard, signature
        log(f"{self.name}: {sum(len(v) for v in by_shard.values())} labelled clips in {len(by_shard)} shards, "
            f"{sum(d for v in by_shard.values() for _, _, d in v) / 3600:.0f} h")

    def list_tars(self) -> None:
        out = subprocess.run([*self.rclone, "lsf", "--format", "ps", "--separator", "\t", self.remote],
                             capture_output=True, text=True)
        if out.returncode:
            log(f"{self.name}: could not list {self.remote}: {out.stderr.strip()[-200:]}")
            return
        sizes = {}
        for line in out.stdout.splitlines():
            name, _, size = line.partition("\t")
            if name.endswith(".tar") and size.isdigit():
                sizes[name[:-4]] = int(size)
        self.sizes = sizes

    def ready(self) -> list[str]:
        return [s for s in self.by_shard if s in self.sizes]

    def next_shard(self, held: set[str], rng: random.Random) -> str | None:
        choices = [s for s in self.ready() if f"{self.name}__{s}" not in held]
        if not choices:
            return None
        least = min(self.visits.get(s, 0) for s in choices)
        return rng.choice(sorted(s for s in choices if self.visits.get(s, 0) == least))

    def hours(self, shard: str) -> float:
        return sum(d for _, _, d in self.by_shard[shard]) / 3600


class Feeder:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.sources: dict[str, Source] = {}
        self.read_config()
        self.pool = Path(self.config["pool"])
        self.pool.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(self.config.get("seed", 0))
        listed = self.pool / "pool.json"
        self.listed: list[str] = json.loads(listed.read_text())["folders"] if listed.exists() else []
        self.listed = [name for name in self.listed if (self.pool / name).is_dir()]
        self.retired: dict[str, float] = {}  # shard folder -> when it left pool.json
        for path in self.pool.iterdir():  # half-unpacked shards, and shards that left the list before a restart
            if path.is_dir() and path.name not in self.listed:
                self.retired[path.name] = 0.0
        self.bytes = {name: folder_bytes(self.pool / name) for name in [*self.listed, *self.retired]}
        self.hours: dict[str, float] = {}
        for name in self.listed:
            manifest = self.pool / name / "manifest.jsonl"
            self.hours[name] = sum(json.loads(l)["duration"] for l in open(manifest)) / 3600 if manifest.exists() else 0
        self.save()
        self.listed_at = 0.0

    def read_config(self) -> None:
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        specs = {spec["name"]: spec for spec in self.config["sources"]}
        for name, spec in specs.items():
            if name in self.sources:
                self.sources[name].update(spec)
            else:
                self.sources[name] = Source(spec, self.config["rclone"])
        for name in [n for n in self.sources if n not in specs]:
            del self.sources[name]

    def relist(self) -> None:
        self.read_config()
        for source in self.sources.values():
            try:
                source.load()
            except (OSError, ValueError, KeyError) as error:
                log(f"{source.name}: cannot read its manifest or labels yet ({error})")
                continue
            source.list_tars()
        self.listed_at = time.time()

    def save(self) -> None:
        write_json(self.pool / "pool.json", {"folders": self.listed, "time": time.time()})

    def held_bytes(self) -> int:
        return sum(self.bytes.get(name, 0) for name in [*self.listed, *self.retired])

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.pool).free

    def delete_released(self) -> None:
        """Delete retired shards that the run's latest inuse.json, written after they left the list, does not hold."""
        try:
            inuse = json.loads((self.pool / "inuse.json").read_text())
        except (OSError, ValueError):
            inuse = None
        for name, since in list(self.retired.items()):
            released = since == 0.0 or (inuse is not None and inuse["time"] > since and name not in inuse["folders"]
                                        and time.time() - inuse["time"] >= GRACE_S)
            if released:
                shutil.rmtree(self.pool / name, ignore_errors=True)
                del self.retired[name]
                self.bytes.pop(name, None)

    def retire_oldest(self) -> None:
        name = self.listed.pop(0)
        self.retired[name] = time.time()
        self.hours.pop(name, None)
        self.save()

    def pick(self) -> tuple[Source, str] | None:
        total = sum(self.hours.values())
        by_source: dict[str, float] = {}
        for name, hours in self.hours.items():
            by_source[name.split("__")[0]] = by_source.get(name.split("__")[0], 0.0) + hours
        weights = sum(s.share for s in self.sources.values() if s.share > 0 and s.ready())
        held = {*self.listed, *self.retired}
        ranked = sorted((s for s in self.sources.values() if s.share > 0 and s.ready()),
                        key=lambda s: by_source.get(s.name, 0.0) / max(total, 1e-9) - s.share / weights)
        for source in ranked:
            shard = source.next_shard(held, self.rng)
            if shard is not None:
                return source, shard
        return None

    def fetch(self, source: Source, shard: str) -> bool:
        name = f"{source.name}__{shard}"
        tmp = self.pool / f".tmp-{name}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        cmd = f"{' '.join(source.rclone)} cat {source.remote}/{shard}.tar | tar -x -C {tmp}"
        started = time.time()
        for attempt in range(3):
            if subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode == 0:
                break
            log(f"{name}: fetch failed (attempt {attempt + 1})")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            time.sleep(20)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        rows = source.by_shard[shard]
        wanted = {json.loads(line)["audio"] for line, _, _ in rows}
        missing = [a for a in wanted if not (tmp / a).exists()]
        for path in (tmp / "clips").rglob("*"):
            if path.is_file() and str(path.relative_to(tmp)) not in wanted:
                path.unlink()
        gone = set(missing)
        keep = [(line, label, d) for line, label, d in rows if json.loads(line)["audio"] not in gone]
        (tmp / "manifest.jsonl").write_text("".join(line for line, _, _ in keep), encoding="utf-8")
        (tmp / f"labels-{self.config.get('labels', 'mix')}.jsonl").write_text("".join(l for _, l, _ in keep),
                                                                             encoding="utf-8")
        tmp.rename(self.pool / name)
        self.bytes[name] = folder_bytes(self.pool / name)
        self.hours[name] = sum(d for _, _, d in keep) / 3600
        self.listed.append(name)
        self.save()
        with open(self.pool / "fetched.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": round(time.time()), "source": source.name, "shard": shard,
                                     "clips": len(keep), "missing": len(missing), "hours": round(self.hours[name], 3),
                                     "bytes": self.bytes[name], "seconds": round(time.time() - started, 1)}) + "\n")
        return True

    def run(self) -> None:
        pool_bytes, floor = self.config["pool_gb"] * 1e9, self.config["floor_gb"] * 1e9
        while not (self.pool / "stop").exists():
            if time.time() - self.listed_at > RELIST_S:
                self.relist()
                pool_bytes, floor = self.config["pool_gb"] * 1e9, self.config["floor_gb"] * 1e9
            self.delete_released()
            choice = self.pick()
            if choice is None:
                time.sleep(30)
                continue
            source, shard = choice
            need = source.sizes[shard]
            if self.held_bytes() + need > pool_bytes or self.free_bytes() - need < floor:
                if self.listed and len(self.retired) < 4 and len(self.listed) > 1:
                    self.retire_oldest()
                time.sleep(5)
                continue
            if self.fetch(source, shard):
                source.visits[shard] = source.visits.get(shard, 0) + 1
                log(f"{source.name}__{shard}: {self.hours[f'{source.name}__{shard}']:.1f} h in; pool "
                    f"{len(self.listed)} shards, {sum(self.hours.values()):.0f} h, {self.held_bytes() / 1e9:.1f} GB "
                    f"held, {self.free_bytes() / 1e9:.1f} GB free")
            else:
                source.visits[shard] = source.visits.get(shard, 0) + 1  # a broken tar must not stall the stream
                log(f"{source.name}__{shard}: could not fetch; skipped")
        log("stop file found; the pool stays as it is")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(__doc__.split("\n\n")[1])
    Feeder(Path(argv[0])).run()


if __name__ == "__main__":
    main()
