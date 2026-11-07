"""Build the deploy package: one folder that runs the akshar server on another machine.

    uv run python scripts/build_deploy.py                    # writes ~/Projects/akshar-deploy
    uv run python scripts/build_deploy.py --zip              # and ~/Projects/akshar-deploy.zip beside it
    uv run python scripts/build_deploy.py /srv/build --models akshar-v0,shruti

The folder holds:
- the API server: akshar.serve and the parts of akshar.eval it uses (no page);
- the .cact files from models/;
- the engine library akshar's scores were measured with (Linux x86-64);
- deploy/'s run script, settings, requirements, service file and README, the README with the models' stored scores.

Unzip it on the server and start ./run.sh there. Rebuild after training a new model. A settings.env or .venv already
in the folder is kept, and .venv never goes into the zip.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from akshar.eval.engine import engine_info  # noqa: E402
from akshar.eval.paths import RUNS_ROOT  # noqa: E402
from akshar.eval.prepare import git_commit  # noqa: E402
from akshar.serve import models as serve_models  # noqa: E402

CODE = ["akshar/__init__.py", "akshar/eval/audio.py", "akshar/eval/engine.py", "akshar/eval/paths.py",
        "akshar/serve/__init__.py", "akshar/serve/__main__.py", "akshar/serve/models.py", "akshar/serve/server.py"]
PACKAGE = ["run.sh", "requirements.txt", "akshar.service", "NOTICE"]
EVAL_INIT = '"""The parts of akshar\'s evaluation harness the server uses: audio loading and the engine\'s identity."""\n'


def scores_table(chosen: list[serve_models.Model]) -> str:
    """The models' stored full-test scores from runs/server-full, with what they depend on (CLAUDE.md, Reporting)."""
    from akshar.ui import results

    table = results.summary(RUNS_ROOT / "server-full")
    by_sha = {row["model_sha256"]: row for row in table["rows"]}
    names = [ts["short"] for ts in table["sets"]]
    lines = ["| Model | File | " + " | ".join(names) + " | Mean WER |", "|---|---|" + "---|" * (len(names) + 1)]
    for model in chosen:
        row = by_sha.get(model.sha256)
        cells = [("–" if row is None or ts["name"] not in row["cells"] else
                  str(row["cells"][ts["name"]]) if ts["name"] == "silence" else f"{row['cells'][ts['name']]:.2f}")
                 for ts in table["sets"]]
        mean = f"{row['mean']:.2f}" if row and row["mean"] is not None else "–"
        lines.append(f"| {model.name} ({model.about}) | {model.bytes / 1e6:.1f} MB | " + " | ".join(cells) + f" | {mean} |")
    hi = ", ".join(ts["short"] for ts in table["sets"] if ts["language"] == "hi" and ts["name"] != "silence")
    en = ", ".join(ts["short"] for ts in table["sets"] if ts["language"] == "en")
    return ("\n".join(lines) + "\n\n"
            "Word error rates (%) on the complete test sets, from akshar's test harness; lower is better. "
            f"Normaliser {', '.join(table['normalisers'])}; {', '.join(table['decoding'])}, needle engine "
            f"{' and '.join(table['engines'])}; language flag hi for {hi}, en for {en}. Monsoon is scored with its "
            "own lattice scorer (OIWER); Silence counts the words written on 100 silent clips. "
            f"Scored on {' and '.join(table['machines'])}. A dash means not scored.")


def build(out: Path, names: list[str] | None) -> dict:
    found = {m.name: m for m in serve_models.find(REPO / "models")}
    missing = [name for name in names or [] if name not in found]
    if missing:
        raise SystemExit(f"no model {', '.join(missing)} under models/; there are {', '.join(found)}")
    chosen = [found[name] for name in names] if names else list(found.values())
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("akshar", "models", "engine"):  # fresh copies; .venv stays
        shutil.rmtree(out / sub, ignore_errors=True)

    for rel in CODE:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, out / rel)
    (out / "akshar/eval/__init__.py").write_text(EVAL_INIT, encoding="utf-8")
    for name in PACKAGE:
        shutil.copy2(REPO / "deploy" / name, out / name)
    shutil.copy2(REPO / "LICENSE", out / "LICENSE")
    (out / "run.sh").chmod(0o755)

    (out / "models").mkdir()
    listed = {}
    for model in chosen:
        shutil.copy2(model.path, out / "models" / model.path.name)
        train = model.path.with_name(f"{model.name}.train.json")
        if train.exists():  # only what the page shows; the full training record stays in the repository
            run = json.loads(train.read_text(encoding="utf-8"))
            (out / "models" / train.name).write_text(json.dumps({k: run[k] for k in ("recipe", "steps") if k in run})
                                                     + "\n", encoding="utf-8")
        listed[model.name] = {"file": f"models/{model.path.name}", "bytes": model.bytes, "sha256": model.sha256}

    import needle

    library = Path(needle._library_path(3))
    engine = engine_info()
    bundled = None
    if sys.platform.startswith("linux") and platform.machine() == "x86_64" and engine["build"] != "unknown":
        bundled = "engine/linux-x86_64/libneedle.so"
        (out / bundled).parent.mkdir(parents=True)
        shutil.copy2(library, out / bundled)

    default = chosen[0].name
    settings = out / "settings.env"
    if not settings.exists():  # an edited settings.env survives rebuilds
        settings.write_text((REPO / "deploy" / "settings.env").read_text(encoding="utf-8").replace("{{default}}", default),
                            encoding="utf-8")
    readme = (REPO / "deploy" / "README.md").read_text(encoding="utf-8").replace("{{default}}", default)
    try:
        table = scores_table(chosen)
    except Exception as error:  # no stored results on this machine
        table = f"No stored test results here ({type(error).__name__}); see docs/akshar.md in the akshar repository."
    (out / "README.md").write_text(readme.replace("<!-- models -->", table), encoding="utf-8")

    record = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "akshar_commit": git_commit(),
              "default_model": default, "models": listed,
              "engine": {"build": engine["build"], "library_sha256": engine["library_sha256"], "bundled": bundled}}
    (out / "BUILD.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def zip_folder(out: Path) -> Path:
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            rel = path.relative_to(out)
            if path.is_dir() or rel.parts[0] == ".venv" or "__pycache__" in rel.parts:
                continue
            zf.write(path, Path(out.name) / rel)  # keeps run.sh executable
    return archive


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", type=Path, default=Path.home() / "Projects" / "akshar-deploy")
    parser.add_argument("--models", help="comma-separated model names (default: every .cact under models/)")
    parser.add_argument("--zip", action="store_true", help="also write <out>.zip, without .venv")
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    if out == REPO or REPO in out.parents:
        raise SystemExit("build the package outside the repository")
    record = build(out, args.models.split(",") if args.models else None)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".venv" not in p.relative_to(out).parts)
    print(f"{out}: {size / 1e6:.1f} MB, models {', '.join(record['models'])} (default {record['default_model']}), "
          f"engine {record['engine']['build']}{' bundled' if record['engine']['bundled'] else ''}")
    if args.zip:
        archive = zip_folder(out)
        print(f"{archive}: {archive.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
