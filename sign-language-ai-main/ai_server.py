"""
수어 인식 AI 서버 (포트 5001)
MediaPipe HandLandmarker + GRU 분류기로 실시간 수어 단어 감지

의존성 설치:
    py -m pip install flask flask-cors mediapipe torch opencv-python pillow

실행:
    cd "sign-language-ai-main"
    py ai_server.py
"""
from __future__ import annotations

import base64
import io
import sys
import time
from collections import Counter, deque
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# sign-language-ai-main 루트를 PYTHONPATH에 추가 (src 임포트용)
sys.path.insert(0, str(Path(__file__).parent))
from src.models.gru_model import GRUSignClassifier
from src.preprocess.extract_landmarks import HandLandmarkExtractor

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
HAND_TASK = BASE_DIR / "sign-language-ai-main" / "models" / "mediapipe" / "hand_landmarker.task"

# 사용할 모델 목록 (좌표 데이터 있는 것만)
MODEL_PATHS = [
    BASE_DIR / "demo_gesture_2026-03-31_v1" / "models" / "best_gru_model.pt",
]

# top30에서 제외할 이상한 레이블
_EXCLUDE_LABELS = {"지시", "지시#", "구형", "너나같다", "등수", "키우다", "한적있다", "방문"}

# ── 추론 하이퍼파라미터 ────────────────────────────────────────────────────────
THRESHOLD     = 0.60   # 5클래스 모델 최적값
STABLE_FRAMES = 3
VOTE_WINDOW   = 6
COOLDOWN      = 4
MIN_TOKEN_GAP = 0.4

import re as _re
def _clean_label(raw: str) -> str:
    s = raw.split(":")[0]
    s = _re.sub(r"[0-9#]+$", "", s)
    return s.strip() or raw

# ── 모델 로드 (전역 1회) ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelEntry:
    def __init__(self, path: Path):
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        m = GRUSignClassifier(
            input_size  = int(ckpt.get("input_size",  126)),
            hidden_size = int(ckpt.get("hidden_size", 128)),
            num_layers  = int(ckpt.get("num_layers",    2)),
            num_classes = int(ckpt["num_classes"]),
            dropout     = float(ckpt.get("dropout", 0.2)),
        ).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        self.model     = m
        self.idx2label = {int(k): _clean_label(str(v)) for k, v in ckpt["idx2label"].items()}
        self.max_len   = int(ckpt.get("max_len",    45))
        self.input_size= int(ckpt.get("input_size", 126))
        # 제외 레이블 필터링
        self.valid_idx = {i for i, lbl in self.idx2label.items() if lbl not in _EXCLUDE_LABELS}

_models: list[ModelEntry] = []
for _p in MODEL_PATHS:
    if _p.exists():
        _me = ModelEntry(_p)
        _models.append(_me)
        print(f"[AI Server] 모델 로드: {_p.name} | {len(_me.idx2label)}클래스")

# 전체 인식 가능 단어 (중복 제거)
_all_labels = sorted(set(
    lbl for m in _models for lbl in m.idx2label.values()
    if lbl not in _EXCLUDE_LABELS
))
_max_len    = max(m.max_len for m in _models)
_input_size = 126

print(f"[AI Server] 총 인식 가능 단어 ({len(_all_labels)}개): {_all_labels}")

# ── 세션 상태 ─────────────────────────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.extractor = HandLandmarkExtractor(
            max_num_hands=2,
            model_asset_path=str(HAND_TASK),
        )
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=_max_len)
        self.recent_labels: deque[str]       = deque(maxlen=VOTE_WINDOW)
        self.candidate_label  = ""
        self.stable_count     = 0
        self.cooldown_count   = 0
        self.last_commit_time = 0.0
        self.lock = Lock()

    def close(self):
        try:
            self.extractor.close()
        except Exception:
            pass


_sessions: dict[str, SessionState] = {}
_sessions_lock = Lock()


def get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = SessionState()
        return _sessions[session_id]


def pad_sequence(sequence: list[np.ndarray], max_len: int, input_size: int) -> np.ndarray:
    if not sequence:
        return np.zeros((1, max_len, input_size), dtype=np.float32)
    arr = np.vstack(sequence).astype(np.float32)
    if len(arr) > max_len:
        arr = arr[-max_len:]
    elif len(arr) < max_len:
        pad = np.zeros((max_len - len(arr), input_size), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return np.expand_dims(arr, axis=0)


# ── Flask 앱 ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.post("/infer")
def infer():
    data       = request.get_json(silent=True) or {}
    session_id = data.get("sessionId", "default")
    frame_b64  = data.get("frameData", "")
    fps        = float(data.get("fps", 5.0))

    if not frame_b64:
        return jsonify({"committedToken": None, "error": "no frameData"})

    # base64 JPEG → numpy BGR (OpenCV 포맷)
    try:
        img_bytes = base64.b64decode(frame_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"committedToken": None, "error": f"decode error: {e}"})

    sess = get_session(session_id)

    with sess.lock:
        # 1. MediaPipe 손 랜드마크 추출 → (126,) 벡터
        feature   = sess.extractor.extract_from_frame(frame, fps=fps)[0]
        has_hands = bool(np.any(np.abs(feature) > 1e-6))
        sess.frame_buffer.append(feature)

        # 2. 멀티모델 GRU 추론 → 가장 높은 신뢰도 채택
        top_label = ""
        top_score = 0.0
        for me in _models:
            seq        = pad_sequence(list(sess.frame_buffer), me.max_len, me.input_size)
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
            with torch.no_grad():
                probs = torch.softmax(me.model(seq_tensor), dim=1)[0]
                # 제외 레이블 마스킹
                for i in range(len(probs)):
                    if i not in me.valid_idx:
                        probs[i] = 0.0
                val, idx = probs.max(dim=0)
                lbl = me.idx2label[int(idx)]
                if float(val) > top_score:
                    top_score = float(val)
                    top_label = lbl

        # 3. 투표 + 안정화 → 단어 커밋
        if has_hands and len(sess.frame_buffer) >= min(8, _max_len) and top_score >= THRESHOLD:
            sess.recent_labels.append(top_label)
        elif sess.recent_labels:
            sess.recent_labels.clear()

        smoothed = ""
        if sess.recent_labels:
            label, votes = Counter(sess.recent_labels).most_common(1)[0]
            if votes >= max(1, VOTE_WINDOW // 2):
                smoothed = label

        committed_token = None

        if smoothed and sess.cooldown_count == 0:
            if smoothed == sess.candidate_label:
                sess.stable_count += 1
            else:
                sess.candidate_label = smoothed
                sess.stable_count    = 1

            if sess.stable_count >= STABLE_FRAMES:
                now = time.time()
                if (now - sess.last_commit_time) >= MIN_TOKEN_GAP:
                    committed_token         = sess.candidate_label
                    sess.last_commit_time   = now
                    print(f"[AI] ✅ 커밋: '{committed_token}'  score={top_score:.2f}  session={session_id[:8]}")
                sess.cooldown_count = COOLDOWN
                sess.stable_count   = 0
                sess.recent_labels.clear()
        else:
            if not has_hands:
                sess.candidate_label = ""
            sess.stable_count = 0

        if sess.cooldown_count > 0:
            sess.cooldown_count -= 1

    return jsonify({
        "committedToken": committed_token,
        "candidate":      sess.candidate_label,
        "score":          round(top_score, 3),
        "hasHands":       has_hands,
    })


@app.delete("/session/<session_id>")
def delete_session(session_id: str):
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].close()
            del _sessions[session_id]
    return jsonify({"ok": True})


@app.get("/health")
def health():
    return jsonify({
        "ok":     True,
        "device": str(device),
        "labels": _all_labels,
    })


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[AI Server] http://0.0.0.0:5001 에서 실행 중...")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
