# sign-language-speech

Flask-based speech API for our sign-language project.

This repository currently provides:

- `TTS`: text to MP3 speech
- `STT`: uploaded audio file to text
- `Realtime STT`: chunk-based speech-to-text session flow
- `Docs page`: Swagger UI plus browser microphone testing

## Stack

- Python
- Flask
- gTTS
- SpeechRecognition
- ffmpeg via `imageio-ffmpeg`
- Swagger UI via `flask-swagger-ui`

## Project structure

- `app.py`: thin entrypoint that creates the Flask app
- `speech_api/__init__.py`: app factory and Swagger blueprint registration
- `speech_api/config.py`: constants and runtime configuration
- `speech_api/openapi.py`: OpenAPI 3 specification builder
- `speech_api/routes.py`: Flask route definitions
- `speech_api/utils.py`: shared response and timing helpers
- `speech_api/services/audio.py`: TTS, STT, ffmpeg conversion, and WAV merge helpers
- `speech_api/services/realtime.py`: realtime STT session state and chunk handling helpers
- `speech_api/state.py`: in-memory realtime session store
- `templates/docs.html`: custom docs page with browser mic test UI
- `requirements.txt`: Python dependencies

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Server:

- `http://127.0.0.1:5000`

## Docs

- Docs page: `http://127.0.0.1:5000/docs`
- Raw Swagger UI: `http://127.0.0.1:5000/swagger`
- OpenAPI JSON: `http://127.0.0.1:5000/openapi.json`

`/docs` is the best entry point for demos because it includes:

- the Swagger UI
- a browser microphone panel
- live transcript preview

## Endpoints

### `GET /health`

Simple health check.

Example response:

```json
{
  "status": "ok",
  "processing_ms": 1
}
```

### `POST /tts`

Convert text to MP3 speech.

Request body:

```json
{
  "text": "Hello from Flask TTS",
  "lang": "en",
  "tld": "com",
  "slow": false
}
```

Example:

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:5000/tts `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"Hello from Flask TTS","lang":"en","tld":"com"}' `
  -OutFile speech.mp3
```

Notes:

- Response body is an MP3 file.
- Processing time is returned in the `X-Processing-Ms` response header.
- Current TTS uses `gTTS`, so internet access is required.

### `POST /stt`

Convert one uploaded audio file to text.

Form fields:

- `audio`
- `language` optional, default `ko-KR`

Supported formats:

- `.wav`
- `.aiff`
- `.aif`
- `.flac`
- `.m4a`
- `.mp3`
- `.mp4`
- `.mpeg`
- `.mpga`
- `.ogg`
- `.webm`

Example:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:5000/stt `
  -Method POST `
  -Form @{
    audio = Get-Item .\sample.m4a
    language = "ko-KR"
  }
```

Example response:

```json
{
  "text": "annyeonghaseyo",
  "language": "ko-KR",
  "filename": "sample.m4a",
  "processing_ms": 842
}
```

Notes:

- Compressed formats are converted to WAV internally before recognition.
- Current STT uses `SpeechRecognition` with Google recognition, so internet access is required.

### Realtime STT flow

Realtime STT is a session-based HTTP flow.

Steps:

1. Start a session
2. Upload audio chunks in order
3. Finish the session and get the final transcript

#### `POST /stt/realtime/start`

Request body:

```json
{
  "language": "ko-KR"
}
```

Example response:

```json
{
  "session_id": "abc123",
  "language": "ko-KR",
  "processing_ms": 3,
  "max_duration_ms": 50000
}
```

#### `POST /stt/realtime/chunk`

Form fields:

- `session_id`
- `sequence_number`
- `audio`
- `language` optional

Example:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:5000/stt/realtime/chunk `
  -Method POST `
  -Form @{
    session_id = $session.session_id
    sequence_number = 1
    audio = Get-Item .\chunk1.wav
  }
```

Example response:

```json
{
  "session_id": "abc123",
  "sequence_number": 1,
  "chunk_text": "hello",
  "accumulated_text": "hello",
  "language": "ko-KR",
  "is_final": false,
  "warning": "",
  "processing_ms": 417,
  "elapsed_ms": 12450,
  "remaining_ms": 37550
}
```

Realtime behavior:

- Chunks must be sent in order.
- `sequence_number` is used to skip duplicate or out-of-order chunks safely.
- Missing or trailing chunks are handled gracefully and can return a warning instead of failing hard.
- Session duration is limited to `50` seconds.

#### `POST /stt/realtime/finish`

Request body:

```json
{
  "session_id": "abc123"
}
```

Example response:

```json
{
  "session_id": "abc123",
  "text": "hello world",
  "language": "ko-KR",
  "is_final": true,
  "processing_ms": 1290,
  "elapsed_ms": 50000
}
```

Notes:

- Chunk responses are optimized for faster feedback.
- Final `finish` tries a fuller pass over the merged session audio for a better final transcript.

## Browser microphone demo

The custom docs page at `/docs` includes a live microphone panel.

What it does:

- requests browser microphone access
- records short WAV chunks in the browser
- sends chunks to `/stt/realtime/chunk`
- shows transcript updates while speaking
- automatically stops at 50 seconds

Recommended browsers:

- Chrome
- Edge

Recommended URLs:

- `http://127.0.0.1:5000/docs`
- `http://localhost:5000/docs`

## Timing fields

Most JSON endpoints include:

- `processing_ms`: server processing time in milliseconds

Realtime endpoints also include:

- `elapsed_ms`: elapsed realtime session duration
- `remaining_ms`: remaining time before the 50-second limit

TTS timing is returned in:

- `X-Processing-Ms` response header

## Current limitations

- TTS depends on an external online service.
- STT currently uses `SpeechRecognition` with Google recognition, which is easy to demo but not the most accurate option.
- Realtime STT is HTTP chunk-based, not WebSocket-based.
- Browser live transcript quality depends on browser support.
- Realtime session state is stored in memory, so it is intended for single-process local demos right now.

## Suggested next steps

- Replace current STT engine with Whisper or faster-whisper
- Add WebSocket-based realtime streaming
- Move session state out of in-memory dictionaries if we need multi-instance deployment
- Add automated API tests for the realtime chunk flow
