"""
TTS / STT 서버 (포트 5000)
Node.js server.js 에서 /tts, /stt, /voices, /docs 로 프록시됨

설치:
    pip install flask flask-cors gtts

실행:
    python tts_server.py
"""

import io
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from gtts import gTTS

app = Flask(__name__)
CORS(app)


# ── TTS ──────────────────────────────────────────────────────────────────────
@app.post("/tts")
def tts():
    """
    요청 body (JSON):
        text : 읽어줄 텍스트
        lang : 언어 코드 (기본 "ko")

    응답:
        audio/mpeg (MP3 바이너리)
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    lang = data.get("lang", "ko")

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        mp3_buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        return Response(mp3_buf.read(), mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── STT (stub) ────────────────────────────────────────────────────────────────
# 앱에서 Android SpeechRecognizer를 직접 사용하므로 현재는 stub
@app.post("/stt")
def stt():
    """
    요청 body: { audio: base64 }  (현재 미사용 — 앱에서 직접 SpeechRecognizer 처리)
    응답:      { text: "" }
    """
    return jsonify({"text": ""})


# ── 보조 엔드포인트 ───────────────────────────────────────────────────────────
@app.get("/voices")
def voices():
    return jsonify({"voices": ["ko", "en", "ja", "zh"]})


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/docs")
def docs():
    return jsonify({
        "endpoints": {
            "POST /tts":    "{ text, lang } → audio/mpeg",
            "POST /stt":    "stub (앱에서 직접 처리)",
            "GET  /voices": "지원 언어 목록",
            "GET  /health": "서버 상태 확인",
        }
    })


# ── 실행 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[TTS Server] http://0.0.0.0:5000 에서 실행 중...")
    app.run(host="0.0.0.0", port=5000, debug=False)
