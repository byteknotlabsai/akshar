"""python -m akshar.ui: a page on this computer for trying akshar's models and reading their test results.

    python -m akshar.ui            # then open http://127.0.0.1:8765

It is the API of akshar.serve with a page on top, and is not part of the deploy package.
- Transcribe: record from the microphone or open an audio file, and run any .cact under models/ on it. Each transcript
  comes with its speed, word timings and confidences; paste what was said to score it with the harness's normaliser.
- Test results: when runs/ and the prepared test sets are here, the full-test tables and every clip behind them.
- API: how an app calls akshar.serve, with this server's address.
"""
from __future__ import annotations

import argparse
import os
import threading
import uuid
import webbrowser
from collections import OrderedDict
from pathlib import Path

from ..eval import audio
from ..eval.normalise import VERSION as NORMALISER_VERSION
from ..serve import models
from ..serve.server import App, ApiError, Handler, Request, Settings, serve
from .align import text_alignment

try:  # the test-set browser needs the harness and the prepared test sets
    from . import results
except ImportError:
    results = None

PAGE = Path(__file__).with_name("page.html")
AUDIO_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}


class Uploads:
    """Clips added on the page, as 16 kHz samples, so that each model runs on one upload. The oldest are dropped once
    more than `limit_mb` is held."""

    def __init__(self, limit_mb: float = 512):
        self._clips: OrderedDict[str, object] = OrderedDict()
        self._limit = limit_mb * 2**20
        self._lock = threading.Lock()

    def add(self, samples) -> str:
        key = uuid.uuid4().hex[:12]
        with self._lock:
            self._clips[key] = samples
            while len(self._clips) > 1 and sum(clip.nbytes for clip in self._clips.values()) > self._limit:
                self._clips.popitem(last=False)
        return key

    def get(self, key: str):
        with self._lock:
            samples = self._clips.get(key)
        if samples is None:
            raise ApiError(404, "that audio is no longer held by the server; add it again")
        return samples


class PageApp(App):
    def __init__(self, model_root: Path, settings: Settings | None = None, show_results: bool = True):
        super().__init__(model_root, settings)
        self.by_sha = {m.sha256: m.name for m in self.models.values()}
        self.results = results if show_results else None
        self.uploads = Uploads()

    def page(self, _query: dict) -> tuple[bytes, str]:
        return PAGE.read_bytes(), "text/html; charset=utf-8"

    def info(self, _query: dict) -> dict:
        settings = self.settings
        return {"models": [m.info() for m in self.models.values()], "default": self.default,
                "default_language": settings.default_language,
                "engine": self.engine["build"], "package": self.engine["package"], "machine": self.machine,
                "threads": os.cpu_count(), "languages": models.LANGUAGES, "normaliser": NORMALISER_VERSION,
                "auth": bool(settings.api_keys), "cors": list(settings.cors_origins),
                "limits": {"max_mb": settings.max_mb, "max_seconds": settings.max_seconds},
                "features": {"results": self.results is not None}}

    def add_audio(self, request: Request) -> dict:
        samples = self.samples(request.body)
        return {"id": self.uploads.add(samples), "seconds": round(len(samples) / audio.SAMPLE_RATE, 2)}

    def transcribe(self, request: Request) -> dict:
        body = request.json()
        model = self.model(body.get("model"))
        if body.get("set"):  # a test clip, scored with its set's scorer
            res = self._results()
            set_name = self._set(body["set"])
            path = res.audio_path(set_name, str(body.get("clip")))
            if path is None:
                raise ApiError(404, "no such clip in that test set")
            language = self.language(body.get("language") or res.sets.SETS[set_name].language)
            result = self.run(model, audio.load(path), language)
            result["score"] = res.alignment(set_name, res.manifest(set_name)[str(body["clip"])], result["text"])
            return result
        result = self.run(model, self.uploads.get(str(body.get("audio"))), self.language(body.get("language")))
        if str(body.get("reference") or "").strip():
            result["score"] = text_alignment(body["reference"], result["text"])
        return result

    def score(self, request: Request) -> dict:
        body = request.json()
        return text_alignment(str(body.get("reference") or ""), str(body.get("hypothesis") or ""))

    # the test-set browser

    def _results(self):
        if self.results is None:
            raise ApiError(404, "test results are not available on this server")
        return self.results

    def _root(self, query: dict) -> Path:
        found = self._results().roots()
        if query.get("root") not in found:
            raise ApiError(404, f"no results root {query.get('root')!r}")
        return found[query["root"]]["path"]

    def _set(self, name) -> str:
        if name not in self._results().sets.SETS:
            raise ApiError(404, f"unknown test set {name!r}")
        return name

    def roots(self, _query: dict) -> list[dict]:
        return [{"name": name, "label": root["label"]} for name, root in self._results().roots().items()]

    def summary(self, query: dict) -> dict:
        table = self._results().summary(self._root(query))
        for row in table["rows"]:
            row["local"] = self.by_sha.get(row["model_sha256"])
        return table

    def clips(self, query: dict) -> dict:
        table = self._results().clip_table(self._root(query), self._set(query.get("set")))
        return {**table, "runs": [{**run, "local": self.by_sha.get(run["model_sha256"])} for run in table["runs"]]}

    def clip(self, query: dict) -> dict:
        detail = self._results().clip_detail(self._root(query), self._set(query.get("set")), str(query.get("id")))
        if detail is None:
            raise ApiError(404, "no such clip in that test set")
        for run in detail["runs"]:
            run["local"] = self.by_sha.get(run["model_sha256"])
        return detail

    def eval_audio(self, query: dict) -> tuple[bytes, str]:
        path = self._results().audio_path(self._set(query.get("set")), str(query.get("id")))
        if path is None:
            raise ApiError(404, "no such clip, or its audio file is missing")
        return path.read_bytes(), AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream")


class PageHandler(Handler):
    OPEN = Handler.OPEN | {"/index.html", "/api/models"}  # the page loads, then asks for a key if one is needed
    GET = {**Handler.GET, "/": "page", "/index.html": "page", "/api/models": "info", "/api/roots": "roots",
           "/api/summary": "summary", "/api/set": "clips", "/api/clip": "clip", "/api/eval-audio": "eval_audio"}
    POST = {**Handler.POST, "/api/audio": "add_audio", "/api/transcribe": "transcribe", "/api/score": "score"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.ui", description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--models", type=Path, default=models.MODELS_ROOT, help="folder searched for .cact files")
    parser.add_argument("--no-results", action="store_true", help="leave out the test-set browser")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)
    app = PageApp(args.models, Settings.from_env(), show_results=not args.no_results)
    if args.host not in ("127.0.0.1", "localhost", "::1") and app.results is not None and app.results.roots():
        print("note: the Test results tab serves the test sets' audio to anyone who can reach this port; "
              "--no-results leaves it out", flush=True)
    if args.open:
        webbrowser.open(f"http://{'127.0.0.1' if args.host in ('', '0.0.0.0', '::') else args.host}:{args.port}")
    serve(app, PageHandler, args.host, args.port)
