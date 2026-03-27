from __future__ import annotations

from typing import Any


def build_openapi_spec() -> dict[str, Any]:
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Sign Language Speech API",
            "version": "1.3.0",
            "description": "Flask API for TTS, file-based STT, and chunk-based realtime STT.",
        },
        "servers": [
            {"url": "http://127.0.0.1:5000", "description": "Local development server"},
        ],
        "tags": [
            {"name": "Health", "description": "Health and status endpoints"},
            {"name": "Voices", "description": "Supported language helper endpoint"},
            {"name": "TTS", "description": "Text-to-speech generation"},
            {"name": "STT", "description": "Speech-to-text recognition from an uploaded file"},
            {"name": "Realtime STT", "description": "Chunk-based realtime speech-to-text session flow"},
        ],
        "paths": {
            "/health": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Health check",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/openapi.json": {
                "get": {
                    "tags": ["Health"],
                    "summary": "OpenAPI specification",
                    "responses": {
                        "200": {
                            "description": "OpenAPI JSON",
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        }
                    },
                }
            },
            "/voices": {
                "get": {
                    "tags": ["Voices"],
                    "summary": "List supported language and audio options",
                    "responses": {
                        "200": {
                            "description": "Supported languages and formats",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/VoicesResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/tts": {
                "post": {
                    "tags": ["TTS"],
                    "summary": "Convert text to speech",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TTSRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Generated MP3 file",
                            "content": {
                                "audio/mpeg": {
                                    "schema": {"type": "string", "format": "binary"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid request",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/stt": {
                "post": {
                    "tags": ["STT"],
                    "summary": "Convert uploaded audio to text",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {"$ref": "#/components/schemas/STTRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Recognized text result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/STTResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid request",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "422": {
                            "description": "Speech could not be recognized",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "502": {
                            "description": "Speech recognition provider error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/stt/realtime/start": {
                "post": {
                    "tags": ["Realtime STT"],
                    "summary": "Start a realtime STT session",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RealtimeStartRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Realtime session created",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RealtimeSessionResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/stt/realtime/chunk": {
                "post": {
                    "tags": ["Realtime STT"],
                    "summary": "Upload one audio chunk",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {"$ref": "#/components/schemas/RealtimeChunkRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Chunk processed or safely skipped",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RealtimeChunkResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid request",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "408": {
                            "description": "Realtime session exceeded its time limit",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "502": {
                            "description": "Speech recognition provider error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/stt/realtime/finish": {
                "post": {
                    "tags": ["Realtime STT"],
                    "summary": "Finish a realtime STT session",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RealtimeFinishRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Realtime session finished",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RealtimeFinishResponse"}
                                }
                            },
                        },
                        "404": {
                            "description": "Session not found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "ok"},
                        "processing_ms": {"type": "integer", "example": 1},
                    },
                    "required": ["status", "processing_ms"],
                },
                "LanguageOption": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "example": "ko"},
                        "name": {"type": "string", "example": "Korean"},
                    },
                    "required": ["code", "name"],
                },
                "VoicesResponse": {
                    "type": "object",
                    "properties": {
                        "languages": {"type": "array", "items": {"$ref": "#/components/schemas/LanguageOption"}},
                        "tlds": {"type": "array", "items": {"type": "string"}},
                        "stt_languages": {"type": "array", "items": {"type": "string"}},
                        "audio_formats": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["languages", "tlds", "stt_languages", "audio_formats"],
                },
                "TTSRequest": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "example": "Hello from Flask TTS"},
                        "lang": {"type": "string", "example": "ko", "default": "ko"},
                        "tld": {"type": "string", "example": "co.kr", "default": "com"},
                        "slow": {"type": "boolean", "default": False},
                    },
                    "required": ["text"],
                },
                "STTRequest": {
                    "type": "object",
                    "properties": {
                        "audio": {"type": "string", "format": "binary"},
                        "language": {"type": "string", "default": "ko-KR", "example": "ko-KR"},
                    },
                    "required": ["audio"],
                },
                "STTResponse": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "example": "annyeonghaseyo"},
                        "language": {"type": "string", "example": "ko-KR"},
                        "filename": {"type": "string", "example": "sample.m4a"},
                        "processing_ms": {"type": "integer", "example": 842},
                    },
                    "required": ["text", "language", "filename", "processing_ms"],
                },
                "RealtimeStartRequest": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string", "example": "ko-KR", "default": "ko-KR"}
                    },
                },
                "RealtimeSessionResponse": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "language": {"type": "string", "example": "ko-KR"},
                        "processing_ms": {"type": "integer", "example": 3},
                        "max_duration_ms": {"type": "integer", "example": 50000},
                    },
                    "required": ["session_id", "language", "processing_ms", "max_duration_ms"],
                },
                "RealtimeChunkRequest": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "sequence_number": {"type": "integer", "minimum": 1, "example": 1},
                        "audio": {"type": "string", "format": "binary"},
                        "language": {"type": "string", "example": "ko-KR"},
                    },
                    "required": ["session_id", "sequence_number", "audio"],
                },
                "RealtimeChunkResponse": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "sequence_number": {"type": "integer", "example": 1},
                        "chunk_text": {"type": "string", "example": "hello"},
                        "accumulated_text": {"type": "string", "example": "hello world"},
                        "language": {"type": "string", "example": "ko-KR"},
                        "is_final": {"type": "boolean", "example": False},
                        "warning": {"type": "string", "example": ""},
                        "processing_ms": {"type": "integer", "example": 417},
                        "elapsed_ms": {"type": "integer", "example": 12450},
                        "remaining_ms": {"type": "integer", "example": 37550},
                    },
                    "required": ["session_id", "sequence_number", "chunk_text", "accumulated_text", "language", "is_final", "warning", "processing_ms", "elapsed_ms", "remaining_ms"],
                },
                "RealtimeFinishRequest": {
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
                "RealtimeFinishResponse": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "text": {"type": "string", "example": "hello world"},
                        "language": {"type": "string", "example": "ko-KR"},
                        "is_final": {"type": "boolean", "example": True},
                        "processing_ms": {"type": "integer", "example": 1290},
                        "elapsed_ms": {"type": "integer", "example": 50000},
                    },
                    "required": ["session_id", "text", "language", "is_final", "processing_ms", "elapsed_ms"],
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string", "example": "audio file is required"},
                        "processing_ms": {"type": "integer", "example": 12},
                    },
                    "required": ["error", "processing_ms"],
                },
            }
        },
    }
