from __future__ import annotations

import time

from flask import Flask, jsonify, render_template, request, send_file

from .config import (
    DEFAULT_LANG,
    DEFAULT_STT_LANG,
    DEFAULT_TLD,
    DOCS_PATH,
    MAX_REALTIME_SESSION_SECONDS,
    MIN_REALTIME_CHUNK_SECONDS,
    OPENAPI_PATH,
    SUPPORTED_STT_EXTENSIONS,
    SWAGGER_UI_PATH,
)
from .services import (
    LANGUAGE_OPTIONS,
    STT_LANGUAGE_OPTIONS,
    TTS_TLDS,
    add_realtime_audio_chunk,
    append_realtime_text,
    build_chunk_response,
    create_realtime_session,
    ensure_wav_audio,
    finish_realtime_session,
    get_realtime_elapsed_ms,
    get_realtime_session,
    get_realtime_text,
    save_uploaded_audio,
    synthesize_to_audio,
    transcribe_audio,
    validate_realtime_sequence,
)
from .utils import elapsed_ms, json_error


def register_routes(app: Flask) -> None:
    @app.get("/health")
    def health():
        start_time = time.perf_counter()
        return jsonify({"status": "ok", "processing_ms": elapsed_ms(start_time)})

    @app.get(OPENAPI_PATH)
    def openapi_spec():
        return jsonify(app.config["OPENAPI_SPEC"])

    @app.get(DOCS_PATH)
    @app.get(f"{DOCS_PATH}/")
    def docs():
        return render_template(
            "docs.html",
            swagger_ui_url=SWAGGER_UI_PATH,
            min_chunk_seconds=MIN_REALTIME_CHUNK_SECONDS,
            max_realtime_duration_ms=MAX_REALTIME_SESSION_SECONDS * 1000,
        )

    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.get("/voices")
    def voices():
        return jsonify(
            {
                "languages": LANGUAGE_OPTIONS,
                "tlds": TTS_TLDS,
                "stt_languages": STT_LANGUAGE_OPTIONS,
                "audio_formats": sorted(SUPPORTED_STT_EXTENSIONS),
            }
        )

    @app.post("/tts")
    def tts():
        start_time = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            return json_error("text is required", 400, start_time)

        lang = str(payload.get("lang") or DEFAULT_LANG)
        tld = str(payload.get("tld") or DEFAULT_TLD)
        slow = bool(payload.get("slow", False))

        try:
            audio_path = synthesize_to_audio(text=text, lang=lang, tld=tld, slow=slow)
        except Exception:
            return json_error("failed to generate speech audio", 502, start_time)

        response = send_file(
            audio_path,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name="speech.mp3",
        )
        response.headers["X-Processing-Ms"] = str(elapsed_ms(start_time))
        return response

    @app.post("/stt")
    def stt():
        start_time = time.perf_counter()
        uploaded_file = request.files.get("audio")
        if uploaded_file is None:
            return json_error("audio file is required", 400, start_time)

        language = str(request.form.get("language") or DEFAULT_STT_LANG)

        try:
            input_path, original_name = save_uploaded_audio(uploaded_file, "stt-input")
            wav_path = ensure_wav_audio(input_path)
            text = transcribe_audio(wav_path, language)
        except ValueError as exc:
            return json_error(str(exc), 400, start_time)
        except LookupError as exc:
            return json_error(str(exc), 422, start_time)
        except ConnectionError as exc:
            return json_error(str(exc), 502, start_time)

        return jsonify(
            {
                "text": text,
                "language": language,
                "filename": original_name,
                "processing_ms": elapsed_ms(start_time),
            }
        )

    @app.post("/stt/realtime/start")
    def realtime_start():
        start_time = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        language = str(payload.get("language") or DEFAULT_STT_LANG)
        session = create_realtime_session(language)
        return jsonify(
            {
                "session_id": session["session_id"],
                "language": session["language"],
                "processing_ms": elapsed_ms(start_time),
                "max_duration_ms": MAX_REALTIME_SESSION_SECONDS * 1000,
            }
        )

    @app.post("/stt/realtime/chunk")
    def realtime_chunk():
        start_time = time.perf_counter()
        session_id = str(request.form.get("session_id") or "").strip()
        sequence_raw = str(request.form.get("sequence_number") or "").strip()
        language = str(request.form.get("language") or DEFAULT_STT_LANG)
        uploaded_file = request.files.get("audio")

        if not session_id:
            return json_error("session_id is required", 400, start_time)
        if not sequence_raw:
            return json_error("sequence_number is required", 400, start_time)
        if uploaded_file is None:
            return json_error("audio file is required", 400, start_time)

        try:
            sequence_number = int(sequence_raw)
        except ValueError:
            return json_error("sequence_number must be an integer", 400, start_time)
        if sequence_number < 1:
            return json_error("sequence_number must be greater than or equal to 1", 400, start_time)

        session = get_realtime_session(session_id)
        if session is None:
            return jsonify(
                build_chunk_response(
                    session_id=session_id,
                    sequence_number=sequence_number,
                    language=language,
                    accumulated_text="",
                    processing_ms=elapsed_ms(start_time),
                    elapsed_ms=0,
                    warning="realtime session not found; chunk skipped",
                )
            )

        elapsed = get_realtime_elapsed_ms(session)
        max_duration_ms = MAX_REALTIME_SESSION_SECONDS * 1000
        if elapsed > max_duration_ms:
            return json_error(
                f"realtime session exceeded the {max_duration_ms} ms limit",
                408,
                start_time,
            )

        language = str(session.get("language") or language)
        warning = validate_realtime_sequence(session, sequence_number)
        if warning:
            return jsonify(
                build_chunk_response(
                    session_id=session_id,
                    sequence_number=sequence_number,
                    language=language,
                    accumulated_text=get_realtime_text(session_id),
                    processing_ms=elapsed_ms(start_time),
                    elapsed_ms=elapsed,
                    warning=warning,
                )
            )

        try:
            input_path, _ = save_uploaded_audio(uploaded_file, "stt-input")
            wav_path = ensure_wav_audio(input_path)
            add_realtime_audio_chunk(session_id, wav_path, sequence_number)
            try:
                chunk_text = transcribe_audio(wav_path, language)
                accumulated_text = append_realtime_text(session_id, chunk_text)
                warning = ""
            except LookupError:
                chunk_text = ""
                accumulated_text = get_realtime_text(session_id)
                warning = "no speech recognized in chunk"
        except ValueError as exc:
            return json_error(str(exc), 400, start_time)
        except KeyError:
            return jsonify(
                build_chunk_response(
                    session_id=session_id,
                    sequence_number=sequence_number,
                    language=language,
                    accumulated_text="",
                    processing_ms=elapsed_ms(start_time),
                    elapsed_ms=0,
                    warning="session already finished; trailing chunk skipped",
                )
            )
        except ConnectionError as exc:
            return json_error(str(exc), 502, start_time)

        refreshed_session = get_realtime_session(session_id)
        refreshed_elapsed = get_realtime_elapsed_ms(refreshed_session) if refreshed_session else elapsed
        return jsonify(
            build_chunk_response(
                session_id=session_id,
                sequence_number=sequence_number,
                language=language,
                accumulated_text=accumulated_text,
                processing_ms=elapsed_ms(start_time),
                elapsed_ms=refreshed_elapsed,
                warning=warning,
                chunk_text=chunk_text,
            )
        )

    @app.post("/stt/realtime/finish")
    def realtime_finish():
        start_time = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            return json_error("session_id is required", 400, start_time)

        try:
            result = finish_realtime_session(session_id)
        except KeyError:
            return json_error("realtime session not found", 404, start_time)

        result["processing_ms"] = elapsed_ms(start_time)
        return jsonify(result)
