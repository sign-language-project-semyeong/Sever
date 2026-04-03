"""
AI 추론 서버 (포트 5001)
Kotlin 앱에서 카메라 프레임을 받아 GRU 모델로 수어 단어를 감지하고
Node.js 서버(/token)로 자동 전송합니다.

흐름:
  Kotlin 앱 → POST /infer (base64 프레임) → 단어 감지 → Node.js /token 전송
  → Node.js가 Gemini 번역 + TTS → WebSocket으로 앱에 오디오 푸시
"""

from __future__ import annotations

import base64
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.gru_model import GRUSignClassifier
from src.preprocess.extract_landmarks import HandLandmarkExtractor

# ── 설정 ───────────────────────────────────────────────────────────────────────
import os
NODE_SERVER_URL   = os.environ.get("NODE_SERVER_URL", "http://localhost:3000")

# 모델 선택: MODEL_NAME=top30, top50, demo / 또는 CHECKPOINT_PATH 직접 지정
_model_name = os.environ.get("MODEL_NAME", "top50").lower()
_checkpoint_override = os.environ.get("CHECKPOINT_PATH", "")

if _checkpoint_override:
    CHECKPOINT_PATH = Path(_checkpoint_override)
elif _model_name == "demo":
    CHECKPOINT_PATH = ROOT.parent.parent / "demo_gesture_2026-03-31_v1" / "models" / "best_gru_model.pt"
elif _model_name == "top30":
    CHECKPOINT_PATH = ROOT / "models" / "checkpoints_top30" / "best_gru_model.pt"
else:
    CHECKPOINT_PATH = ROOT / "models" / "checkpoints_top50" / "best_gru_model.pt"

MODEL_ASSET_PATH  = ROOT / "models" / "mediapipe" / "hand_landmarker.task"
print(f"[AI Server] 사용 모델: {_model_name} ({CHECKPOINT_PATH})")

THRESHOLD      = 0.5   # 0.7 → 0.5 (더 넓게 인식)
STABLE_FRAMES  = 5    # 8 → 5 (더 빠르게 확정)
VOTE_WINDOW    = 10
COOLDOWN_FRAMES = 10
MIN_TOKEN_GAP  = 1.5   # 같은 단어 재등록 최소 간격(초)

app = Flask(__name__)
CORS(app)

# ── 모델 로드 ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[AI Server] device: {device}")

def load_model(path: Path):
    ckpt = torch.load(path, map_location=device)
    m = GRUSignClassifier(
        input_size  = int(ckpt.get("input_size", 126)),
        hidden_size = int(ckpt.get("hidden_size", 128)),
        num_layers  = int(ckpt.get("num_layers", 2)),
        num_classes = int(ckpt["num_classes"]),
        dropout     = float(ckpt.get("dropout", 0.2)),
    ).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    idx2label = {int(k): str(v) for k, v in ckpt["idx2label"].items()}
    max_len    = int(ckpt.get("max_len", 30))
    input_size = int(ckpt.get("input_size", 126))
    return m, idx2label, max_len, input_size

model, idx2label, max_len, input_size = load_model(CHECKPOINT_PATH)
extractor = HandLandmarkExtractor(
    max_num_hands=2, model_asset_path=str(MODEL_ASSET_PATH)
)
_extractor_lock = threading.Lock()  # MediaPipe는 멀티스레드 비안전 → 락 필요
print("[AI Server] 모델 로드 완료")

# ── 세션 상태 ──────────────────────────────────────────────────────────────────
SESSION_TIMEOUT = 300  # 5분 동안 요청 없으면 세션 삭제

_sessions: dict[str, dict] = {}
_lock = threading.Lock()

def _new_session() -> dict:
    return {
        "frame_buffer"   : deque(maxlen=max_len),
        "recent_labels"  : deque(maxlen=VOTE_WINDOW),
        "current_tokens" : [],
        "candidate_label": "",
        "stable_count"   : 0,
        "cooldown_count" : 0,
        "last_token_time": time.time(),
        "last_commit_time": 0.0,
        "last_active"    : time.time(),
    }

def get_session(sid: str) -> dict:
    with _lock:
        if sid not in _sessions:
            _sessions[sid] = _new_session()
        _sessions[sid]["last_active"] = time.time()
        return _sessions[sid]

def _cleanup_sessions() -> None:
    """만료된 세션 주기적으로 삭제 (5분마다)"""
    while True:
        time.sleep(300)
        now = time.time()
        with _lock:
            expired = [sid for sid, s in _sessions.items()
                       if now - s.get("last_active", 0) > SESSION_TIMEOUT]
            for sid in expired:
                del _sessions[sid]
                print(f"[AI Server] 만료 세션 삭제: {sid}")

threading.Thread(target=_cleanup_sessions, daemon=True).start()

# ── 헬퍼 ───────────────────────────────────────────────────────────────────────
def pad_sequence(seq: list, ml: int, isz: int) -> np.ndarray:
    if not seq:
        return np.zeros((1, ml, isz), dtype=np.float32)
    arr = np.vstack(seq).astype(np.float32)
    if len(arr) > ml:
        arr = arr[-ml:]
    elif len(arr) < ml:
        arr = np.vstack([np.zeros((ml - len(arr), isz), dtype=np.float32), arr])
    return np.expand_dims(arr, 0)

def predict_top1(seq_tensor: torch.Tensor) -> tuple[str, float]:
    with torch.no_grad():
        probs = torch.softmax(model(seq_tensor), dim=1)[0]
        val, idx = torch.max(probs, dim=0)
    return idx2label[int(idx)], float(val)

def smooth(history: deque, min_votes: int) -> str:
    if not history:
        return ""
    label, votes = Counter(history).most_common(1)[0]
    return label if votes >= min_votes else ""

def _send_token(sid: str, token: str) -> None:
    """백그라운드에서 Node.js /token 호출"""
    try:
        requests.post(
            f"{NODE_SERVER_URL}/token",
            json={"sessionId": sid, "token": token},
            timeout=3,
        )
        print(f"[AI Server] 토큰 전송 → Node.js: {token}")
    except Exception as e:
        print(f"[AI Server] Node.js 전송 실패: {e}")

# ── API ────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({"ok": True, "device": str(device)})

@app.post("/infer")
def infer():
    """
    요청 body (JSON):
      sessionId : 세션 식별자 (Kotlin 앱이 생성)
      frameData : 카메라 프레임 (base64 JPEG/PNG)
      fps       : 카메라 FPS (기본 30)

    응답 (JSON):
      sessionId     : 동일 세션 ID
      hasHands      : 손 감지 여부
      currentTokens : 현재 누적된 단어 목록
      committedToken: 이번 프레임에서 새로 확정된 단어 (없으면 null)
    """
    data = request.get_json(silent=True) or {}
    sid       = data.get("sessionId", "")
    frame_b64 = data.get("frameData", "")
    fps       = float(data.get("fps", 30.0))

    if not sid or not frame_b64:
        return jsonify({"error": "sessionId and frameData are required"}), 400

    # 프레임 디코딩 (최대 5MB 제한)
    MAX_FRAME_BYTES = 5 * 1024 * 1024
    try:
        frame_bytes = base64.b64decode(frame_b64)
        if len(frame_bytes) > MAX_FRAME_BYTES:
            return jsonify({"error": "frameData too large (max 5MB)"}), 413
        frame_arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame       = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        return jsonify({"error": f"Invalid frameData: {e}"}), 400

    s = get_session(sid)

    # 랜드마크 추출 (MediaPipe 멀티스레드 비안전 → 락으로 보호)
    with _extractor_lock:
        frame_feature = extractor.extract_from_frame(frame, fps=fps)[0]
    has_hands = bool(np.any(np.abs(frame_feature) > 1e-6))
    s["frame_buffer"].append(frame_feature)

    committed_token = None

    # 버퍼가 max_len 이상이면 추론
    if len(s["frame_buffer"]) >= max_len:
        seq_np     = pad_sequence(list(s["frame_buffer"]), max_len, input_size)
        seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device)
        top_label, top_score = predict_top1(seq_tensor)

        if has_hands and top_score >= THRESHOLD:
            s["recent_labels"].append(top_label)
            print(f"[INFER] OK {top_label} ({top_score:.2f}) stable={s['stable_count']} cool={s['cooldown_count']}", flush=True)
        elif has_hands:
            print(f"[INFER] LOW {top_label} ({top_score:.2f}) < threshold {THRESHOLD}", flush=True)
            if s["recent_labels"]:
                s["recent_labels"].clear()
        elif s["recent_labels"]:
            s["recent_labels"].clear()

        smoothed = smooth(s["recent_labels"], max(1, VOTE_WINDOW // 2))

        if smoothed and s["cooldown_count"] == 0:
            if smoothed == s["candidate_label"]:
                s["stable_count"] += 1
            else:
                s["candidate_label"] = smoothed
                s["stable_count"]    = 1

            if s["stable_count"] >= STABLE_FRAMES:
                now = time.time()
                tokens = s["current_tokens"]
                if (now - s["last_commit_time"]) >= MIN_TOKEN_GAP and (
                    not tokens or tokens[-1] != s["candidate_label"]
                ):
                    committed_token = s["candidate_label"]
                    tokens.append(committed_token)
                    s["last_token_time"]  = now
                    s["last_commit_time"] = now
                    # Node.js에 비동기 전송
                    threading.Thread(
                        target=_send_token, args=(sid, committed_token), daemon=True
                    ).start()

                s["cooldown_count"] = COOLDOWN_FRAMES
                s["stable_count"]   = 0
                s["recent_labels"].clear()
        else:
            if not has_hands:
                s["candidate_label"] = ""
            s["stable_count"] = 0

        if s["cooldown_count"] > 0:
            s["cooldown_count"] -= 1

    return jsonify({
        "sessionId"     : sid,
        "hasHands"      : has_hands,
        "currentTokens" : s["current_tokens"],
        "committedToken": committed_token,
    })

@app.delete("/session/<sid>")
def delete_session(sid: str):
    with _lock:
        _sessions.pop(sid, None)
    return jsonify({"ok": True, "sessionId": sid})

# ── 실행 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[AI Server] http://0.0.0.0:5001 에서 실행 중...")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
