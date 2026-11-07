"""python -m akshar.serve: akshar's speech-to-text API, for apps. The deploy package runs it (deploy/README.md).

    python -m akshar.serve --host 0.0.0.0 --port 8765

Endpoints:
    POST /v1/transcribe   the recording as the "file" field of a form, or as the whole request body. Optional fields,
                          as form fields or query parameters: model, language, keywords, words. Answers with JSON.
    GET  /v1/models       the models this server offers, and its defaults
    GET  /health          200 while the server is up; needs no key

Settings come from the environment (the deploy package's settings.env sets them):
    AKSHAR_MODELS            models to offer, comma-separated (default: every .cact under --models)
    AKSHAR_DEFAULT_MODEL     the model for requests that name none (default: the first akshar model)
    AKSHAR_DEFAULT_LANGUAGE  the language flag for requests that give none: hi, en or auto (default hi)
    AKSHAR_API_KEYS          comma-separated keys; when set, requests need "Authorization: Bearer <key>"
                             or "X-API-Key: <key>"
    AKSHAR_CORS_ORIGINS      web origins allowed to call the API from a browser, comma-separated (default *, any)
    AKSHAR_MAX_MB            largest upload in MB (default 100)
    AKSHAR_MAX_SECONDS       longest recording in seconds (default 600)
    AKSHAR_MAX_WAITING       requests that may wait for the engine before the server answers 503 (default 8)
"""
from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import re
import threading
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..eval import audio
from ..eval.engine import cpu_name, engine_info
from . import models

LOOPBACK = ("127.0.0.1", "localhost", "::1")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Settings:
    models: tuple[str, ...] = ()
    default_model: str | None = None
    default_language: str = "hi"
    api_keys: tuple[str, ...] = ()
    cors_origins: tuple[str, ...] = ("*",)
    max_mb: float = 100.0
    max_seconds: float = 600.0
    max_waiting: int = 8

    @classmethod
    def from_env(cls, env=os.environ) -> Settings:
        def items(name: str, default: str = "") -> tuple[str, ...]:
            return tuple(item.strip() for item in (env.get(name) or default).split(",") if item.strip())

        language = env.get("AKSHAR_DEFAULT_LANGUAGE") or "hi"
        if language not in models.LANGUAGES:
            raise SystemExit(f"AKSHAR_DEFAULT_LANGUAGE must be one of {', '.join(models.LANGUAGES)}, not {language!r}")
        return cls(models=items("AKSHAR_MODELS"), default_model=env.get("AKSHAR_DEFAULT_MODEL") or None,
                   default_language=language, api_keys=items("AKSHAR_API_KEYS"),
                   cors_origins=items("AKSHAR_CORS_ORIGINS", "*"), max_mb=float(env.get("AKSHAR_MAX_MB") or 100),
                   max_seconds=float(env.get("AKSHAR_MAX_SECONDS") or 600),
                   max_waiting=int(env.get("AKSHAR_MAX_WAITING") or 8))


def decode(blob: bytes):
    """16 kHz mono samples from encoded audio. WAV, FLAC, MP3 and OGG are read exactly as the harness reads them
    (akshar.eval.audio). Other formats, such as the WebM and MP4 that browsers record and the M4A of phones, are decoded
    with PyAV where it is installed (the deploy package installs it), then mixed and resampled the same way."""
    try:
        return audio.load(io.BytesIO(blob))
    except Exception:  # not a format libsndfile reads
        pass
    try:
        import av
    except ImportError:
        raise ApiError(415, "this server reads WAV, FLAC, MP3 and OGG; install PyAV (pip install av) for other formats, "
                            "or convert the audio first") from None
    import numpy as np
    import soxr

    chunks, rate = [], audio.SAMPLE_RATE
    try:
        with av.open(io.BytesIO(blob)) as container:
            stream = container.streams.audio[0]
            planar = av.AudioResampler(format="fltp")  # float samples, keeping the stream's channels and rate
            for frame in container.decode(stream):
                for out in planar.resample(frame):
                    rate = out.sample_rate
                    chunks.append(out.to_ndarray())
            chunks.extend(out.to_ndarray() for out in planar.resample(None))
    except Exception:
        raise ApiError(415, "the audio could not be decoded") from None
    if not chunks:
        return np.zeros(0, np.float32)
    mono = np.concatenate(chunks, axis=1).mean(axis=0, dtype=np.float32)
    if rate != audio.SAMPLE_RATE and len(mono):
        mono = soxr.resample(mono, rate, audio.SAMPLE_RATE, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def multipart(content_type: str, body: bytes) -> tuple[dict[str, str], bytes | None]:
    """The text fields and the file of a multipart/form-data body, as a browser's FormData or `curl -F` sends it."""
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise ApiError(400, "the form has no boundary")
    delimiter = b"\r\n--" + match.group(1).encode("latin-1")
    fields, blob = {}, None
    for part in (b"\r\n" + body).split(delimiter)[1:]:  # each part follows a line break and the delimiter
        if part.startswith(b"--"):  # the closing delimiter
            break
        head, _, data = part.partition(b"\r\n\r\n")
        disposition = next((line for line in head.decode("utf-8", "replace").split("\r\n")
                            if line.lower().startswith("content-disposition:")), "")
        name = re.search(r'\bname="([^"]*)"', disposition)
        if not name:
            continue
        if "filename=" in disposition:
            if blob is None or name.group(1) == "file":
                blob = data
        else:
            fields[name.group(1)] = data.decode("utf-8", "replace")
    return fields, blob


@dataclass
class Request:
    query: dict
    body: bytes
    content_type: str

    def json(self) -> dict:
        try:
            return json.loads(self.body)
        except ValueError:
            raise ApiError(400, "the request body is not JSON") from None

    def form(self) -> tuple[dict, bytes | None]:
        """The fields and the recording: from a multipart form, or from the query string and the whole body."""
        if self.content_type.lower().startswith("multipart/form-data"):
            fields, blob = multipart(self.content_type, self.body)
            return {**self.query, **fields}, blob
        return dict(self.query), self.body


def keyword_list(value) -> list[str] | None:
    """Keywords from a field: words or phrases separated by commas or new lines."""
    keywords = [k.strip() for k in re.split(r"[,\n]", value or "") if k.strip()]
    if len(keywords) > 100 or sum(map(len, keywords)) > 2000:
        raise ApiError(400, "keywords: at most 100 of them, 2,000 characters in all")
    return keywords or None


def flag(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    if str(value).lower() in ("1", "true", "yes", "on"):
        return True
    if str(value).lower() in ("0", "false", "no", "off"):
        return False
    raise ApiError(400, f"expected true or false, not {value!r}")


class App:
    def __init__(self, model_root: Path, settings: Settings | None = None):
        self.settings = settings or Settings()
        found = {m.name: m for m in models.find(model_root)} if model_root.is_dir() else {}
        wanted = self.settings.models or tuple(found)
        missing = [name for name in wanted if name not in found]
        if missing:
            raise SystemExit(f"AKSHAR_MODELS names {', '.join(missing)}, which {model_root} lacks; "
                             f"it has {', '.join(found) or 'none'}")
        self.models = {name: found[name] for name in wanted}
        self.default = self.settings.default_model or next(iter(self.models), None)
        if self.default is not None and self.default not in self.models:
            raise SystemExit(f"AKSHAR_DEFAULT_MODEL={self.default} is not one of the models: {', '.join(self.models)}")
        self.engine = engine_info()
        self.machine = cpu_name()
        self._waiting = 0
        self._waiting_lock = threading.Lock()

    def model(self, name) -> models.Model:
        name = name or self.default
        if name not in self.models:
            raise ApiError(404, f"no model {name!r}; this server has {', '.join(self.models) or 'none'}")
        return self.models[name]

    def language(self, value) -> str:
        value = value or self.settings.default_language
        if value not in models.LANGUAGES:
            raise ApiError(400, f"language must be one of {', '.join(models.LANGUAGES)}")
        return value

    def samples(self, blob: bytes | None):
        if not blob:
            raise ApiError(400, "the request has no audio")
        samples = decode(blob)
        seconds = len(samples) / audio.SAMPLE_RATE
        if not len(samples):
            raise ApiError(400, "the audio is empty")
        if seconds > self.settings.max_seconds:
            raise ApiError(413, f"the audio is {seconds:.0f} s long; this server takes up to "
                                f"{self.settings.max_seconds:.0f} s")
        return samples

    def run(self, model: models.Model, samples, language: str, keywords=None, words: bool = True) -> dict:
        """One decode. Requests queue for the engine; past max_waiting of them the server answers 503."""
        with self._waiting_lock:
            if self._waiting >= self.settings.max_waiting:
                raise ApiError(503, f"the server is busy with {self._waiting} requests; try again shortly")
            self._waiting += 1
        try:
            result = models.transcribe(model, samples, language, keywords, words)
        finally:
            with self._waiting_lock:
                self._waiting -= 1
        print(f"{model.name} ({language}): {result['audio_s']:.1f} s of audio in {result['decode_s']:.2f} s",
              flush=True)
        return result

    # endpoints

    def index(self, _query: dict) -> dict:
        return {"service": "akshar speech to text",
                "endpoints": ["POST /v1/transcribe", "GET /v1/models", "GET /health"]}

    def health(self, _query: dict) -> dict:
        return {"status": "ok", "models": list(self.models), "engine": self.engine["build"]}

    def v1_models(self, _query: dict) -> dict:
        return {"default_model": self.default, "default_language": self.settings.default_language,
                "languages": list(models.LANGUAGES), "models": [m.info() for m in self.models.values()]}

    def v1_transcribe(self, request: Request) -> dict:
        fields, blob = request.form()
        model = self.model(fields.get("model"))
        language = self.language(fields.get("language"))
        keywords = keyword_list(fields.get("keywords"))
        return self.run(model, self.samples(blob), language, keywords, flag(fields.get("words"), True))


class Handler(BaseHTTPRequestHandler):
    """Routes requests to App methods: GET ones take the query, POST ones a Request. Subclasses add routes."""

    app: App
    server_version = "akshar"
    timeout = 120  # seconds a connection may sit idle, so that stalled clients do not hold threads forever
    OPEN = {"/", "/health"}  # every other path needs a key when keys are set
    GET = {"/": "index", "/health": "health", "/v1/models": "v1_models"}
    POST = {"/v1/transcribe": "v1_transcribe"}

    def do_OPTIONS(self):  # a browser's preflight before a cross-origin POST
        self._respond(204, b"", None, {"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                                       "Access-Control-Allow-Headers": "Authorization, Content-Type, X-API-Key",
                                       "Access-Control-Max-Age": "600"})

    def do_GET(self):
        url = urlparse(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        self._answer(url.path, lambda: self._get(url.path, query))

    def do_POST(self):
        url = urlparse(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        self._answer(url.path, lambda: self._post(url.path, query))

    def _get(self, path: str, query: dict):
        if path not in self.GET:
            raise ApiError(404, f"nothing at {path}")
        return getattr(self.app, self.GET[path])(query)

    def _post(self, path: str, query: dict):
        if path not in self.POST:
            raise ApiError(404, f"nothing at {path}")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ApiError(400, "the request has no body")
        if length > self.app.settings.max_mb * 2**20:
            raise ApiError(413, f"uploads are limited to {self.app.settings.max_mb:g} MB")
        request = Request(query, self.rfile.read(length), self.headers.get("Content-Type", ""))
        return getattr(self.app, self.POST[path])(request)

    def _authorised(self) -> bool:
        keys = self.app.settings.api_keys
        if not keys:
            return True
        header = self.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else self.headers.get("X-API-Key", "")
        return any(hmac.compare_digest(token.encode(), key.encode()) for key in keys)

    def _answer(self, path: str, handle):
        extra = {}
        try:
            if path not in self.OPEN and not self._authorised():
                extra["WWW-Authenticate"] = "Bearer"
                raise ApiError(401, "this server needs an API key: send Authorization: Bearer <key>")
            result, status = handle(), 200
        except ApiError as error:
            result, status = {"error": str(error)}, error.status
            if status == 503:
                extra["Retry-After"] = "5"
        except (Exception, SystemExit) as error:  # SystemExit: the harness's way of refusing, in the page's routes
            traceback.print_exc()
            result, status = {"error": f"{type(error).__name__}: {error}"}, 500
        if isinstance(result, tuple):  # (bytes, content type), for pages and audio files
            body, content_type = result
        else:
            body, content_type = json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"
        self._respond(status, body, content_type, extra)

    def _respond(self, status: int, body: bytes, content_type: str | None, headers: dict):
        try:
            self.send_response(status)
            origin, allowed = self.headers.get("Origin"), self.app.settings.cors_origins
            if origin and "*" in allowed:
                self.send_header("Access-Control-Allow-Origin", "*")
            elif origin in allowed:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            if content_type:
                self.send_header("Content-Type", content_type)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # the client went away mid-answer
            pass

    def log_message(self, format, *args):  # each decode prints its own line; other requests stay quiet
        pass


def serve(app: App, handler: type[Handler], host: str, port: int) -> None:
    """Load the default model, listen, and serve until Ctrl+C."""
    if not app.models:
        raise SystemExit("no .cact models to serve")
    models.load(app.models[app.default])  # a missing or incompatible engine shows now, not at the first request
    try:
        server = ThreadingHTTPServer((host, port), type("BoundHandler", (handler,), {"app": app}))
    except OSError as error:
        raise SystemExit(f"cannot listen on {host}:{port} ({error.strerror}); try another port") from None
    settings = app.settings
    shown = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    print(f"akshar on http://{shown}:{port}{'' if host in LOOPBACK else ' and every network address of this machine'}"
          f" (Ctrl+C stops it)", flush=True)
    print(f"  models: {', '.join(app.models)}; defaults: model {app.default}, language {settings.default_language}; "
          f"engine {app.engine['build']}")
    print(f"  API key: {'required' if settings.api_keys else 'not required'}; browsers may call it from "
          f"{'any origin' if '*' in settings.cors_origins else ', '.join(settings.cors_origins)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m akshar.serve", description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on; 0.0.0.0 for every network "
                                                            "(default: this computer only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--models", type=Path, default=models.MODELS_ROOT, help="folder searched for .cact files")
    args = parser.parse_args(argv)
    serve(App(args.models, Settings.from_env()), Handler, args.host, args.port)
