const fs = require("fs");
const path = require("path");
const http = require("http");
const dotenv = require("dotenv");
const express = require("express");
const cors = require("cors");
const { WebSocketServer } = require("ws");
const { GoogleGenAI } = require("@google/genai");
const { createProxyMiddleware } = require("http-proxy-middleware");

dotenv.config();

// ── API Key ────────────────────────────────────────────────────────────────────
function loadGeminiApiKey() {
  if (process.env.GEMINI_API_KEY) return process.env.GEMINI_API_KEY.trim();
  if (process.env.GOOGLE_API_KEY) return process.env.GOOGLE_API_KEY.trim();

  const envPath = path.join(__dirname, ".env");
  if (!fs.existsSync(envPath)) return "";

  const rawEnv = fs.readFileSync(envPath, "utf8").trim();
  if (!rawEnv) return "";

  const firstLine = rawEnv.split(/\r?\n/)[0].trim();
  if (firstLine.includes("=")) {
    const [, value = ""] = firstLine.split("=");
    return value.trim().replace(/^['"]|['"]$/g, "");
  }
  return firstLine;
}

function normalizeGlossInput(glosses) {
  if (Array.isArray(glosses)) {
    return glosses.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof glosses === "string") {
    return glosses.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean);
  }
  return [];
}

const apiKey = loadGeminiApiKey();
if (!apiKey) {
  throw new Error(
    "Gemini API key not found. Add GEMINI_API_KEY=your_key to .env, or keep the key as the first line."
  );
}

const ai = new GoogleGenAI({ apiKey });
const app = express();
const port = process.env.PORT || 3000;
const defaultModel = process.env.GEMINI_MODEL || "gemini-2.5-flash-lite";
const cacheTtlMs = Number(process.env.CACHE_TTL_MS || 30000);
const responseCache = new Map();

// ── CORS ───────────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ── Python STT/TTS 서버 프록시 (포트 5000) ────────────────────────────────────
const PYTHON_SERVER_URL = process.env.PYTHON_SERVER_URL || "http://localhost:5000";

async function forwardToPython(req, res, path) {
  try {
    const isJson = req.headers["content-type"]?.includes("application/json");
    const fetchRes = await fetch(`${PYTHON_SERVER_URL}${path}`, {
      method: req.method,
      headers: { "Content-Type": req.headers["content-type"] || "application/json" },
      body: isJson ? JSON.stringify(req.body) : undefined,
    });
    const contentType = fetchRes.headers.get("content-type") || "";
    res.status(fetchRes.status);
    res.set("Content-Type", contentType);
    const buf = await fetchRes.arrayBuffer();
    res.send(Buffer.from(buf));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}

app.post("/tts", (req, res) => forwardToPython(req, res, "/tts"));
app.post("/stt", (req, res) => forwardToPython(req, res, "/stt"));
app.get("/voices", (req, res) => forwardToPython(req, res, "/voices"));
app.get("/docs", (req, res) => forwardToPython(req, res, "/docs"));

// ── AI 추론 서버 프록시 (포트 5001) ──────────────────────────────────────────
// Kotlin 앱 → POST /infer → AI 서버(수어 감지) → Node.js /token 자동 전송
const AI_SERVER_URL = process.env.AI_SERVER_URL || "http://localhost:5001";
app.post("/infer", async (req, res) => {
  try {
    const response = await fetch(`${AI_SERVER_URL}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await response.json();
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── 세션 버퍼 ──────────────────────────────────────────────────────────────────
const sessionBuffers = new Map();
const BUFFER_FLUSH_MS = Number(process.env.BUFFER_FLUSH_MS || 2000);

function getOrCreateSession(sessionId) {
  if (!sessionBuffers.has(sessionId)) {
    sessionBuffers.set(sessionId, { tokens: [], timer: null });
  }
  return sessionBuffers.get(sessionId);
}

function clearSession(sessionId) {
  const session = sessionBuffers.get(sessionId);
  if (session?.timer) clearTimeout(session.timer);
  sessionBuffers.delete(sessionId);
}

// ── Gemini 번역 ────────────────────────────────────────────────────────────────
async function buildSentenceFromGlosses(glosses) {
  const cacheKey = glosses.join(" ").toLowerCase();
  const cached = responseCache.get(cacheKey);

  if (cached && cached.expiresAt > Date.now()) {
    return { sentence: cached.sentence, cached: true };
  }

  const prompt = [
    "You turn sign-language gloss tokens into one natural Korean sentence.",
    "Return only the sentence.",
    `Input glosses: ${glosses.join(" ")}`
  ].join("\n");

  const config = { maxOutputTokens: 80, temperature: 0.2 };
  if (defaultModel.startsWith("gemini-2.5")) {
    config.thinkingConfig = { thinkingBudget: 0 };
  }

  const response = await ai.models.generateContent({
    model: defaultModel,
    contents: prompt,
    config
  });

  const sentence = response.text.trim();
  responseCache.set(cacheKey, { sentence, expiresAt: Date.now() + cacheTtlMs });
  return { sentence, cached: false };
}

// ── Python TTS 호출 → 오디오 버퍼 반환 ──────────────────────────────────────
function callPythonTtsBuffer(sentence) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ text: sentence, lang: "ko" });
    const ttsUrl = new URL(`${PYTHON_SERVER_URL}/tts`);
    const options = {
      hostname: ttsUrl.hostname,
      port: ttsUrl.port || 5000,
      path: "/tts",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
      },
    };
    const req = http.request(options, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => resolve(Buffer.concat(chunks)));
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

// ── HTTP Routes ────────────────────────────────────────────────────────────────
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

// 수어 단어 → 문장 번역 → TTS 오디오 반환 (체인 엔드포인트)
app.post("/gloss-to-speech", async (req, res) => {
  const glosses = normalizeGlossInput(req.body.glosses);
  if (glosses.length === 0) {
    return res.status(400).json({ error: "glosses is required." });
  }

  let sentence;
  try {
    const result = await buildSentenceFromGlosses(glosses);
    sentence = result.sentence;
  } catch (error) {
    console.error("Translation error:", error);
    return res.status(500).json({ error: "Failed to translate glosses." });
  }

  const ttsBody = JSON.stringify({ text: sentence, lang: "ko" });
  const ttsUrl = new URL(`${PYTHON_SERVER_URL}/tts`);
  const options = {
    hostname: ttsUrl.hostname,
    port: ttsUrl.port || 5000,
    path: "/tts",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(ttsBody),
    },
  };

  const ttsReq = http.request(options, (ttsRes) => {
    if (ttsRes.statusCode !== 200) {
      return res.status(502).json({ error: "TTS failed.", sentence });
    }
    res.set("Content-Type", "audio/mpeg");
    res.set("X-Sentence", encodeURIComponent(sentence));
    ttsRes.pipe(res);
  });

  ttsReq.on("error", (err) => {
    console.error("TTS request error:", err);
    res.status(502).json({ error: "Python TTS server unreachable.", sentence });
  });

  ttsReq.write(ttsBody);
  ttsReq.end();
});

// 기존 일괄 번역 엔드포인트
app.post("/translate", async (req, res) => {
  try {
    const glosses = normalizeGlossInput(req.body.glosses);
    if (glosses.length === 0) {
      return res.status(400).json({
        error: "glosses is required."
      });
    }

    const startedAt = Date.now();
    const result = await buildSentenceFromGlosses(glosses);
    return res.json({
      glosses,
      sentence: result.sentence,
      cached: result.cached,
      elapsedMs: Date.now() - startedAt,
      model: defaultModel
    });
  } catch (error) {
    console.error("Translation error:", error);
    return res.status(500).json({ error: "Failed to generate sentence with Gemini." });
  }
});

// 단어 단위 버퍼 엔드포인트 - AI 서버에서 단어 감지 시 자동 호출
// POST /token { sessionId: "...", token: "person" }
app.post("/token", (req, res) => {
  const { sessionId, token } = req.body;

  if (!sessionId || !token) {
    return res.status(400).json({ error: "sessionId and token are required." });
  }

  const session = getOrCreateSession(sessionId);
  session.tokens.push(String(token).trim());

  if (session.timer) clearTimeout(session.timer);
  session.timer = setTimeout(() => {
    flushSession(sessionId);
  }, BUFFER_FLUSH_MS);

  // WebSocket으로 현재 버퍼 상태 실시간 전송
  const ws = wsClients.get(sessionId);
  if (ws && ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify({ type: "buffered", tokens: session.tokens }));
  }

  return res.json({
    sessionId,
    buffered: session.tokens,
    message: "Token received. Translation will trigger automatically."
  });
});

// 즉시 번역 요청
app.post("/flush", async (req, res) => {
  const { sessionId } = req.body;
  if (!sessionId) {
    return res.status(400).json({ error: "sessionId is required." });
  }

  const session = sessionBuffers.get(sessionId);
  if (!session || session.tokens.length === 0) {
    return res.status(400).json({ error: "No tokens in buffer." });
  }

  if (session.timer) clearTimeout(session.timer);

  const glosses = [...session.tokens];
  clearSession(sessionId);

  try {
    const startedAt = Date.now();
    const result = await buildSentenceFromGlosses(glosses);
    return res.json({
      sessionId,
      glosses,
      sentence: result.sentence,
      cached: result.cached,
      elapsedMs: Date.now() - startedAt,
      model: defaultModel
    });
  } catch (error) {
    console.error("Flush translation error:", error);
    return res.status(500).json({ error: "Failed to generate sentence with Gemini." });
  }
});

app.delete("/session/:sessionId", (req, res) => {
  clearSession(req.params.sessionId);
  res.json({ ok: true, sessionId: req.params.sessionId });
});

// ── WebSocket ──────────────────────────────────────────────────────────────────
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

const wsClients = new Map();

// 연결된 모든 청취자(listener-*)에게 메시지 브로드캐스트
function broadcastToListeners(payload) {
  const data = JSON.stringify(payload);
  for (const [sid, client] of wsClients.entries()) {
    if (sid.startsWith("listener-") && client.readyState === client.OPEN) {
      client.send(data);
    }
  }
}

// 타이머 만료 시 자동 번역 → TTS → WebSocket으로 오디오 푸시
async function flushSession(sessionId) {
  const session = sessionBuffers.get(sessionId);
  if (!session || session.tokens.length === 0) return;

  const glosses = [...session.tokens];
  clearSession(sessionId);

  const ws = wsClients.get(sessionId);

  try {
    const result = await buildSentenceFromGlosses(glosses);
    const sentence = result.sentence;

    const sentencePayload = {
      type: "sentence", sessionId, glosses, sentence,
      cached: result.cached, model: defaultModel
    };

    // 1. 번역 문장 → 수어 사용자 앱 + 모든 청취자
    if (ws && ws.readyState === ws.OPEN) ws.send(JSON.stringify(sentencePayload));
    broadcastToListeners(sentencePayload);

    // 2. TTS 오디오 → 수어 사용자 앱 + 모든 청취자
    try {
      const audioBuffer = await callPythonTtsBuffer(sentence);
      const audioPayload = {
        type: "audio", sessionId, sentence,
        audio: audioBuffer.toString("base64"), mimeType: "audio/mpeg"
      };
      if (ws && ws.readyState === ws.OPEN) ws.send(JSON.stringify(audioPayload));
      broadcastToListeners(audioPayload);
    } catch (ttsError) {
      console.error("TTS error:", ttsError);
      if (ws && ws.readyState === ws.OPEN)
        ws.send(JSON.stringify({ type: "error", error: "TTS failed.", sentence }));
    }
  } catch (error) {
    console.error("Auto-flush error:", error);
  }
}

wss.on("connection", (ws) => {
  let clientSessionId = null;

  ws.on("message", async (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      ws.send(JSON.stringify({ type: "error", error: "Invalid JSON." }));
      return;
    }

    const { type, sessionId, token, glosses } = msg;

    if (type === "join") {
      clientSessionId = sessionId;
      wsClients.set(sessionId, ws);
      ws.send(JSON.stringify({ type: "joined", sessionId }));
      return;
    }

    if (type === "stt") {
      // 보낸 사람 제외한 모든 세션에 릴레이
      for (const [sid, client] of wsClients.entries()) {
        if (sid !== clientSessionId && client.readyState === client.OPEN) {
          client.send(JSON.stringify({ type: "stt", text: msg.text, from: clientSessionId }));
        }
      }
      return;
    }

    if (type === "token") {
      if (!clientSessionId) {
        ws.send(JSON.stringify({ type: "error", error: "Send { type: 'join', sessionId } first." }));
        return;
      }
      const session = getOrCreateSession(clientSessionId);
      session.tokens.push(String(token).trim());

      if (session.timer) clearTimeout(session.timer);
      session.timer = setTimeout(() => flushSession(clientSessionId), BUFFER_FLUSH_MS);

      ws.send(JSON.stringify({ type: "buffered", tokens: session.tokens }));
      return;
    }

    if (type === "flush") {
      const sid = clientSessionId || sessionId;
      if (!sid) {
        ws.send(JSON.stringify({ type: "error", error: "No sessionId." }));
        return;
      }
      const session = sessionBuffers.get(sid);
      if (!session || session.tokens.length === 0) {
        ws.send(JSON.stringify({ type: "error", error: "Buffer is empty." }));
        return;
      }
      if (session.timer) clearTimeout(session.timer);
      const toTranslate = [...session.tokens];
      clearSession(sid);

      try {
        const result = await buildSentenceFromGlosses(toTranslate);
        ws.send(JSON.stringify({
          type: "sentence",
          sessionId: sid,
          glosses: toTranslate,
          sentence: result.sentence,
          cached: result.cached,
          model: defaultModel
        }));
      } catch (err) {
        ws.send(JSON.stringify({ type: "error", error: "Translation failed." }));
      }
      return;
    }

    if (type === "translate") {
      const normalized = normalizeGlossInput(glosses);
      if (normalized.length === 0) {
        ws.send(JSON.stringify({ type: "error", error: "glosses is empty." }));
        return;
      }
      try {
        const result = await buildSentenceFromGlosses(normalized);
        ws.send(JSON.stringify({
          type: "sentence",
          glosses: normalized,
          sentence: result.sentence,
          cached: result.cached,
          model: defaultModel
        }));
      } catch (err) {
        ws.send(JSON.stringify({ type: "error", error: "Translation failed." }));
      }
      return;
    }

    ws.send(JSON.stringify({ type: "error", error: `Unknown type: ${type}` }));
  });

  ws.on("close", () => {
    if (clientSessionId) {
      wsClients.delete(clientSessionId);
      clearSession(clientSessionId);
    }
  });
});

server.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
  console.log(`WebSocket ready on ws://localhost:${port}`);
});
