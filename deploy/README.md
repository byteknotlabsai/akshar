# akshar server

Speech to text for Hindi, English and Hinglish, as an HTTP API that runs on the server's CPU. Your frontend sends a recording, and the server answers with the transcript as JSON. There is no page: just the API.

## Start it

```bash
unzip akshar-deploy.zip && cd akshar-deploy
./run.sh
```

The first start installs the Python packages into `.venv`, which needs internet and takes a minute or two. Check that it is up:

```bash
curl http://localhost:8765/health
```

It needs Linux on x86-64 with Python 3.9 or newer, or [uv](https://docs.astral.sh/uv/), which fetches Python itself. Other systems (macOS, ARM Linux) work too: there the engine package downloads its own engine build on the first start.

## Settings

All settings are in `settings.env`. Edit it and restart the server.

| Setting | Default | What it does |
|---|---|---|
| `AKSHAR_HOST` | 0.0.0.0 | The address to listen on: every network, or 127.0.0.1 for this machine only. |
| `AKSHAR_PORT` | 8765 | The port. |
| `AKSHAR_MODELS` | all | The models to offer, comma-separated, from `models/`. |
| `AKSHAR_DEFAULT_MODEL` | {{default}} | The model for requests that name none. |
| `AKSHAR_DEFAULT_LANGUAGE` | hi | The language for requests that give none: `hi`, `en` or `auto`. |
| `AKSHAR_API_KEYS` | none | Comma-separated keys. When set, requests need `Authorization: Bearer <key>`. |
| `AKSHAR_CORS_ORIGINS` | `*` | The websites whose pages may call the API, e.g. `https://app.example.com`. |
| `AKSHAR_MAX_MB` | 100 | The largest upload, in MB. |
| `AKSHAR_MAX_SECONDS` | 600 | The longest recording, in seconds. |
| `AKSHAR_MAX_WAITING` | 8 | Requests that may queue before the server answers 503. |
| `AKSHAR_CPUS` | all | CPUs for the engine, e.g. `12-23`, on a machine shared with other busy services. |

A variable set in the environment wins over `settings.env`, e.g. `AKSHAR_PORT=9000 ./run.sh`.

## The API

### POST /v1/transcribe

Transcribes one recording. Send it as the `file` field of a form (multipart/form-data), or as the whole request body. The other fields are optional:

| Field | Default | Meaning |
|---|---|---|
| `file` | | The recording: WAV, MP3, M4A, WebM, OGG, FLAC, MP4 and other common formats. |
| `model` | {{default}} | Which model; `GET /v1/models` lists them. |
| `language` | hi | `hi` for Hindi and Hinglish, `en` for English, `auto` to let the model decide. |
| `keywords` | | Words or phrases to favour, separated by commas: names, product terms. |
| `words` | true | `false` leaves out the word timings. |

When the recording is the request body, the other fields go in the query string: `/v1/transcribe?language=en&model=shruti`.

From your frontend, with a file from an `<input type="file">` or the Blob from a `MediaRecorder`:

```js
export async function transcribe(recording, {language = "hi", model, keywords} = {}) {
  const form = new FormData();
  form.append("file", recording);
  form.append("language", language);
  if (model) form.append("model", model);
  if (keywords) form.append("keywords", keywords.join(","));
  const response = await fetch("https://akshar.example.com/v1/transcribe", {method: "POST", body: form});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  return result;  // result.text is the transcript
}
```

When `AKSHAR_API_KEYS` is set, add `headers: {Authorization: "Bearer <key>"}` to the `fetch`.

From a terminal:

```bash
curl http://localhost:8765/v1/transcribe -F file=@voice-note.m4a -F language=hi -F keywords="Hassan,Hoysala"
```

The answer (values illustrative):

```json
{
  "text": "तुम्हारा laptop अच्छा लग रहा है",
  "model": "{{default}}",
  "language": "hi",
  "language_flag": "hi",
  "audio_s": 3.12,
  "decode_s": 0.061,
  "x_real_time": 51.1,
  "words": [{"word": "तुम्हारा", "start": 0.0, "end": 0.56, "probability": 0.97}, ...]
}
```

- `text` is the transcript in the models' mixed style: Hindi in Devanagari, English words in Latin letters, numbers as words, no punctuation.
- `language` is the language the model used: the one asked for, or the one it detected with `auto`. `language_flag` is what the request asked for.
- `audio_s` is the recording's length and `decode_s` the time the model took, in seconds.
- `words` gives each word's start and end in seconds and how sure the model was, from 0 to 1.

Errors come back as `{"error": "..."}`:

| Status | Meaning |
|---|---|
| 400 | Bad field, or no audio |
| 401 | Missing or wrong key |
| 404 | No such model |
| 413 | Recording too large or too long |
| 415 | Recording that can't be read |
| 503 | Busy; retry in a few seconds |

### GET /v1/models

The models this server offers and its defaults:

```json
{"default_model": "{{default}}", "default_language": "hi", "languages": ["hi", "en", "auto"],
 "models": [{"name": "{{default}}", "file": "models/{{default}}.cact", "mb": 29.5, "about": "...", "sha256": "..."}, ...]}
```

### GET /health

Answers `{"status": "ok", ...}` while the server is up, and needs no key.

## Calling it from a website

- **CORS:** a page on another address may call the API only if `AKSHAR_CORS_ORIGINS` allows its origin. The default `*` allows any; set it to your frontend's address, e.g. `https://app.example.com`.
- **HTTPS:** a page served over `https://` cannot call an `http://` API, so give the API an `https://` address with a reverse proxy, as below.
- **Keys:** anyone can read a key in a public website's JavaScript. If the server is reachable from the internet, call it from your website's backend with the key, or keep it on a private network.

With [Caddy](https://caddyserver.com), which gets the certificate itself:

```
akshar.example.com {
    reverse_proxy 127.0.0.1:8765
}
```

With nginx, inside the `server` block that has your certificate:

```
location / {
    proxy_pass http://127.0.0.1:8765;
    client_max_body_size 100m;   # nginx's default of 1 MB would refuse most recordings
    proxy_read_timeout 300s;
}
```

Behind a proxy, set `AKSHAR_HOST=127.0.0.1` so that the server is reachable only through the proxy.

## Keep it running

With systemd, edit the user and paths in `akshar.service` and follow the steps at its top. The server then restarts if it fails and starts at boot. For a quick trial, `tmux new -s akshar ./run.sh` also works.

## Speed and memory

The engine decodes one request at a time and uses most of the machine's cores; other requests wait their turn. On the laptop this package was built on (Intel Core i5-12500H), a 10-second recording takes about 0.2 s, roughly 50 times faster than real time. The server uses about 120 MB of memory.

On a machine shared with other busy services, give the engine its own CPUs with `AKSHAR_CPUS`. Its speed drops sharply when its threads share CPUs with another busy process.

## Models

<!-- models -->

## Updating

Build a new folder from the akshar repository, copy it over and keep your `settings.env`:

```bash
uv run python scripts/build_deploy.py --zip
```

`BUILD.json` records what each build holds: the repository commit, and every model's and the engine's SHA-256.

## Licences

- **Code:** Apache-2.0 (`LICENSE`).
- **Engine:** Cactus Compute's needle engine, Apache-2.0.
- **Shruti:** Udgam Labs' model, Apache-2.0.
- **akshar v0 and akshar-lite v0:** start from Cactus Whistle (Apache-2.0), trained on the datasets `NOTICE` lists, with Shruti as their teacher. Every one of those sources is admitted in the akshar project's licence ledger. MUCS 2021, the share-alike one, was cleared for commercial use by the owner's legal guidance (decision D26).

`NOTICE` lists every component and dataset.
