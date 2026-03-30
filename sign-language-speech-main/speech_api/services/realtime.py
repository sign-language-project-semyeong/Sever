from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import (
    DEFAULT_STT_LANG,
    MAX_REALTIME_SESSION_SECONDS,
    RECENTLY_CLOSED_SESSION_GRACE_SECONDS,
    TEMP_ROOT,
)
from ..state import REALTIME_LOCK, REALTIME_SESSIONS, RECENTLY_CLOSED_REALTIME_SESSIONS
from .audio import append_wav_audio, transcribe_audio


def _prune_recently_closed_sessions(now: float | None = None) -> None:
    current_time = now or time.time()
    expired_ids = [
        session_id
        for session_id, closed_at in RECENTLY_CLOSED_REALTIME_SESSIONS.items()
        if current_time - closed_at > RECENTLY_CLOSED_SESSION_GRACE_SECONDS
    ]
    for session_id in expired_ids:
        RECENTLY_CLOSED_REALTIME_SESSIONS.pop(session_id, None)


def create_realtime_session(language: str = DEFAULT_STT_LANG) -> dict[str, Any]:
    session_id = uuid4().hex
    session_dir = TEMP_ROOT / f"realtime-stt-{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    session = {
        "session_id": session_id,
        "language": language,
        "directory": session_dir,
        "audio_path": session_dir / "session.wav",
        "transcript": "",
        "started_at": time.perf_counter(),
        "last_sequence_number": 0,
    }
    with REALTIME_LOCK:
        REALTIME_SESSIONS[session_id] = session
        RECENTLY_CLOSED_REALTIME_SESSIONS.pop(session_id, None)
        _prune_recently_closed_sessions()
    return session


def get_realtime_session(session_id: str) -> dict[str, Any] | None:
    with REALTIME_LOCK:
        _prune_recently_closed_sessions()
        session = REALTIME_SESSIONS.get(session_id)
        if session is None:
            return None
        return dict(session)


def get_realtime_elapsed_ms(session: dict[str, Any]) -> int:
    return int((time.perf_counter() - float(session["started_at"])) * 1000)


def validate_realtime_sequence(session: dict[str, Any], sequence_number: int) -> str | None:
    last_sequence_number = int(session.get("last_sequence_number", 0))
    if sequence_number <= last_sequence_number:
        return "duplicate or old chunk skipped"
    if sequence_number != last_sequence_number + 1:
        return "out-of-order chunk skipped"
    return None


def append_realtime_text(session_id: str, chunk_text: str) -> str:
    with REALTIME_LOCK:
        session = REALTIME_SESSIONS.get(session_id)
        if session is None:
            return ""
        existing_text = str(session.get("transcript", "")).strip()
        chunk_text = chunk_text.strip()
        combined = (
            f"{existing_text} {chunk_text}".strip()
            if existing_text and chunk_text
            else existing_text or chunk_text
        )
        session["transcript"] = combined
        return combined


def get_realtime_text(session_id: str) -> str:
    with REALTIME_LOCK:
        session = REALTIME_SESSIONS.get(session_id)
        if session is None:
            return ""
        return str(session.get("transcript", ""))


def add_realtime_audio_chunk(session_id: str, wav_chunk_path: Path, sequence_number: int) -> dict[str, Any]:
    with REALTIME_LOCK:
        session = REALTIME_SESSIONS.get(session_id)
        if session is None:
            raise KeyError(session_id)
        audio_path = Path(session["audio_path"])
        append_wav_audio(audio_path, wav_chunk_path)
        session["last_sequence_number"] = sequence_number
        return dict(session)


def finish_realtime_session(session_id: str) -> dict[str, Any]:
    with REALTIME_LOCK:
        session = REALTIME_SESSIONS.get(session_id)
        if session is None:
            raise KeyError(session_id)
        session_copy = dict(session)

    final_text = str(session_copy.get("transcript", "")).strip()
    audio_path = Path(session_copy["audio_path"])
    if audio_path.exists():
        try:
            recognized_text = transcribe_audio(audio_path, str(session_copy.get("language", DEFAULT_STT_LANG)))
            if recognized_text:
                final_text = recognized_text
        except (LookupError, ConnectionError, ValueError):
            pass

    elapsed = get_realtime_elapsed_ms(session_copy)
    language = str(session_copy.get("language", DEFAULT_STT_LANG))

    with REALTIME_LOCK:
        REALTIME_SESSIONS.pop(session_id, None)
        RECENTLY_CLOSED_REALTIME_SESSIONS[session_id] = time.time()
        _prune_recently_closed_sessions()

    return {
        "session_id": session_id,
        "text": final_text,
        "language": language,
        "is_final": True,
        "elapsed_ms": min(elapsed, MAX_REALTIME_SESSION_SECONDS * 1000),
    }


def build_chunk_response(
    session_id: str,
    sequence_number: int,
    language: str,
    accumulated_text: str,
    processing_ms: int,
    elapsed_ms: int,
    warning: str = "",
    chunk_text: str = "",
) -> dict[str, Any]:
    max_duration_ms = MAX_REALTIME_SESSION_SECONDS * 1000
    return {
        "session_id": session_id,
        "sequence_number": sequence_number,
        "chunk_text": chunk_text,
        "accumulated_text": accumulated_text,
        "language": language,
        "is_final": False,
        "warning": warning,
        "processing_ms": processing_ms,
        "elapsed_ms": elapsed_ms,
        "remaining_ms": max(max_duration_ms - elapsed_ms, 0),
    }
