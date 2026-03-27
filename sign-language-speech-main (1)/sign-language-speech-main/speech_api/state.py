from __future__ import annotations

import threading

REALTIME_SESSIONS: dict[str, dict[str, object]] = {}
RECENTLY_CLOSED_REALTIME_SESSIONS: dict[str, float] = {}
REALTIME_LOCK = threading.Lock()
