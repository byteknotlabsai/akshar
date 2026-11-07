import io
import json
import random
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from akshar.serve import models, server

HAVE_MODELS = any(models.MODELS_ROOT.rglob("*.cact"))


def _serve(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("TestHandler", (server.Handler,), {"app": app}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _running(app):
    httpd, url = _serve(app)
    yield url
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def keyed_api(tmp_path):
    settings = server.Settings(api_keys=("team-key",), cors_origins=("https://app.example",))
    yield from _running(server.App(tmp_path / "no-models", settings))


@pytest.fixture
def model_api():
    yield from _running(server.App(models.MODELS_ROOT, server.Settings(max_seconds=5)))


def _get(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as response:
            return response.status, json.loads(response.read()), response.headers
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), error.headers


def _post(url, data, headers=None):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _form(fields: dict, files: dict, boundary: str = "----akshar-test-boundary") -> tuple[bytes, str]:
    body = b""
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode() + value.encode() + b"\r\n"
    for name, (filename, data) in files.items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def _wav(seconds: float) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(16000 * seconds), np.float32), 16000, format="WAV")
    return buffer.getvalue()


def test_forms_keep_binary_files_intact():
    rng = random.Random(1)
    blob = bytes(rng.randrange(256) for _ in range(5000)) + b"\r\n--\r\n\r\n--x" + b"\r\n"
    body, content_type = _form({"language": "en", "keywords": "राम, Hassan"}, {"file": ("a.wav", blob)})
    fields, data = server.multipart(content_type, body)
    assert data == blob
    assert fields == {"language": "en", "keywords": "राम, Hassan"}


def test_settings_from_the_environment():
    settings = server.Settings.from_env({"AKSHAR_API_KEYS": " k1, k2 ,", "AKSHAR_CORS_ORIGINS": "https://a.example",
                                         "AKSHAR_MAX_SECONDS": "60", "AKSHAR_MODELS": "akshar-v0",
                                         "AKSHAR_DEFAULT_LANGUAGE": "en", "AKSHAR_MAX_MB": ""})
    assert settings.api_keys == ("k1", "k2") and settings.cors_origins == ("https://a.example",)
    assert settings.max_seconds == 60 and settings.max_mb == 100 and settings.max_waiting == 8
    assert settings.models == ("akshar-v0",) and settings.default_language == "en"
    assert server.Settings.from_env({}).cors_origins == ("*",)
    with pytest.raises(SystemExit):
        server.Settings.from_env({"AKSHAR_DEFAULT_LANGUAGE": "fr"})


def test_request_fields():
    assert server.keyword_list("राम, Hassan\nHoysala dynasty,,") == ["राम", "Hassan", "Hoysala dynasty"]
    assert server.keyword_list("") is None
    with pytest.raises(server.ApiError):
        server.keyword_list(",".join(["x"] * 101))
    assert server.flag(None, True) is True and server.flag("false", True) is False and server.flag("1", False) is True
    with pytest.raises(server.ApiError):
        server.flag("maybe", True)


def test_unknown_models_stop_the_server(tmp_path):
    with pytest.raises(SystemExit):
        server.App(tmp_path, server.Settings(models=("akshar-v9",)))


def test_keys_and_allowed_origins(keyed_api):
    api = keyed_api
    assert _get(api + "/health")[0] == 200  # open, for monitoring
    assert _get(api + "/")[0] == 200
    assert _get(api + "/v1/models")[0] == 401
    assert _get(api + "/v1/models", {"Authorization": "Bearer wrong"})[0] == 401
    assert _get(api + "/v1/models", {"Authorization": "Bearer team-key"})[0] == 200
    assert _get(api + "/v1/models", {"X-API-Key": "team-key"})[0] == 200
    assert _get(api + "/health", {"Origin": "https://app.example"})[2]["Access-Control-Allow-Origin"] == "https://app.example"
    assert "Access-Control-Allow-Origin" not in _get(api + "/health", {"Origin": "https://other.example"})[2]
    assert _get(api + "/api/models", {"Authorization": "Bearer team-key"})[0] == 404  # no page routes here


@pytest.mark.skipif(not HAVE_MODELS, reason="no .cact model under models/")
def test_transcribe_end_to_end(model_api):
    api = model_api
    body, content_type = _form({"language": "hi", "keywords": "राम"}, {"file": ("silence.wav", _wav(1))})
    status, result = _post(api + "/v1/transcribe", body, {"Content-Type": content_type})
    assert status == 200 and result["text"].strip() == "" and result["audio_s"] == 1.0 and "words" in result
    status, result = _post(api + "/v1/transcribe?language=auto&words=false", _wav(1))  # the audio as the body
    assert status == 200 and result["language_flag"] == "auto" and "words" not in result
    assert _post(api + "/v1/transcribe", _wav(6))[0] == 413
    assert _post(api + "/v1/transcribe", b"not audio")[0] == 415
    assert _post(api + "/v1/transcribe?model=nope", _wav(1))[0] == 404
    assert _post(api + "/v1/transcribe?language=fr", _wav(1))[0] == 400
    status, listing, _ = _get(api + "/v1/models")
    assert status == 200 and listing["default_language"] == "hi"
    assert listing["default_model"] == listing["models"][0]["name"]
