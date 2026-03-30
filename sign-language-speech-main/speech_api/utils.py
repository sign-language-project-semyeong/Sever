from __future__ import annotations

import time
from typing import Any

from flask import jsonify


class ApiError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def json_error(message: str, status_code: int, start_time: float, **extra: Any):
    payload: dict[str, Any] = {"error": message, "processing_ms": elapsed_ms(start_time)}
    payload.update(extra)
    return jsonify(payload), status_code
