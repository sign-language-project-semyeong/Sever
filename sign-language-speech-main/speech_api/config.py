from __future__ import annotations

import os
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe

DEFAULT_LANG = "ko"
DEFAULT_TLD = "com"
DEFAULT_STT_LANG = "ko-KR"

OPENAPI_PATH = "/openapi.json"
DOCS_PATH = "/docs"
SWAGGER_UI_PATH = "/swagger"

DIRECT_STT_EXTENSIONS = {".wav", ".aiff", ".aif", ".flac"}
CONVERTIBLE_STT_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".webm",
}
SUPPORTED_STT_EXTENSIONS = DIRECT_STT_EXTENSIONS | CONVERTIBLE_STT_EXTENSIONS

TEMP_ROOT = Path(".runtime-temp")
TEMP_ROOT.mkdir(exist_ok=True)

MIN_REALTIME_CHUNK_SECONDS = 0.8
MAX_REALTIME_SESSION_SECONDS = 50
RECENTLY_CLOSED_SESSION_GRACE_SECONDS = 5

FFMPEG_EXE = get_ffmpeg_exe()
FFMPEG_DIR = str(Path(FFMPEG_EXE).parent)
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
