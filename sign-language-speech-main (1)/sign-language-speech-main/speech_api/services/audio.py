from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from uuid import uuid4

import speech_recognition as sr
from gtts import gTTS
from werkzeug.datastructures import FileStorage

from ..config import (
    CONVERTIBLE_STT_EXTENSIONS,
    DEFAULT_LANG,
    DEFAULT_STT_LANG,
    DEFAULT_TLD,
    FFMPEG_EXE,
    SUPPORTED_STT_EXTENSIONS,
    TEMP_ROOT,
)

LANGUAGE_OPTIONS = [
    {"code": "ko", "name": "Korean"},
    {"code": "en", "name": "English"},
    {"code": "ja", "name": "Japanese"},
]
TTS_TLDS = ["com", "co.kr", "com.au", "co.jp"]
STT_LANGUAGE_OPTIONS = ["ko-KR", "en-US", "ja-JP"]


def synthesize_to_audio(
    text: str,
    lang: str = DEFAULT_LANG,
    tld: str = DEFAULT_TLD,
    slow: bool = False,
) -> Path:
    output_dir = TEMP_ROOT / f"tts-output-{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "speech.mp3"
    tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
    tts.save(str(output_path))
    return output_path


def save_uploaded_audio(uploaded_file: FileStorage, prefix: str) -> tuple[Path, str]:
    original_name = uploaded_file.filename or "audio"
    suffix = Path(original_name).suffix.lower()
    work_dir = TEMP_ROOT / f"{prefix}-{uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / f"input{suffix or '.bin'}"
    uploaded_file.save(input_path)
    return input_path, original_name


def convert_audio_to_wav(source_path: Path) -> Path:
    target_path = source_path.with_suffix(".wav")
    command = [
        FFMPEG_EXE,
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not target_path.exists():
        raise ValueError(
            "failed to decode audio file; make sure the uploaded file is a valid supported audio file"
        )
    return target_path


def ensure_wav_audio(source_path: Path) -> Path:
    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_STT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_STT_EXTENSIONS))
        raise ValueError(f"unsupported audio format; use one of: {allowed}")
    if suffix in CONVERTIBLE_STT_EXTENSIONS:
        return convert_audio_to_wav(source_path)
    return source_path


def append_wav_audio(target_path: Path, chunk_path: Path) -> None:
    if not target_path.exists():
        shutil.copyfile(chunk_path, target_path)
        return

    with wave.open(str(target_path), "rb") as target_wav:
        params = target_wav.getparams()
        target_frames = target_wav.readframes(target_wav.getnframes())

    with wave.open(str(chunk_path), "rb") as chunk_wav:
        chunk_params = chunk_wav.getparams()
        chunk_frames = chunk_wav.readframes(chunk_wav.getnframes())

    if params[:4] != chunk_params[:4]:
        raise ValueError("realtime audio chunk format did not match the active session audio")

    with wave.open(str(target_path), "wb") as merged_wav:
        merged_wav.setparams(params)
        merged_wav.writeframes(target_frames)
        merged_wav.writeframes(chunk_frames)


def transcribe_audio(audio_path: Path, language: str = DEFAULT_STT_LANG) -> str:
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(str(audio_path)) as source:
            audio = recognizer.record(source)
    except Exception as exc:
        raise ValueError(
            "failed to decode audio file; make sure the uploaded file is a valid supported audio file"
        ) from exc

    try:
        return recognizer.recognize_google(audio, language=language).strip()
    except sr.UnknownValueError as exc:
        raise LookupError("speech was recognized but no text could be extracted") from exc
    except sr.RequestError as exc:
        raise ConnectionError("speech recognition provider error") from exc
