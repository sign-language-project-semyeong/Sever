const fs = require("fs");
const path = require("path");
const http = require("http");
const dotenv = require("dotenv");
const express = require("express");
const cors = require("cors");
const { WebSocketServer } = require("ws");
const { GoogleGenAI } = require("@google/genai");

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
// 코틀린 앱 및 모든 출처 허용 (배포 후 origin을 앱 도메인으로 제한 권장)
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ── 세션 버퍼 ──────────────────────────────────────────────────────────────────
// sessionId → { tokens: string[], timer: Timeout | null }
const sessionBuffers = new Map();
const BUFFER_FLUSH_MS = Number(process.env.BUFFER_FLUSH_MS || 2000); // 마지막 단어 후 2초 뒤 자동 번역

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

// ── HTTP Routes ────────────────────────────────────────────────────────────────
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

// 기존 일괄 번역 엔드포인트 (유지)
app.post("/translate", async (req, res) => {
  try {
    const glosses = normalizeGlossInput(req.body.glosses);
    if (glosses.length === 0) {
      return res.status(400).json({
        error: "glosses is required. Send an array like ['person', 'eat', 'apple'] or a string."
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

// 단어 단위 버퍼 엔드포인트 - 코틀린 앱에서 단어가 인식될 때마다 호출
// POST /token { sessionId: "...", token: "person" }
app.post("/token", (req, res) => {
  const { sessionId, token } = req.body;

  if (!sessionId || !token) {
    return res.status(400).json({ error: "sessionId and token are required." });
  }

  const session = getOrCreateSession(sessionId);
  session.tokens.push(String(token).trim());

  // 기존 타이머 리셋 (마지막 단어 후 BUFFER_FLUSH_MS 뒤 자동 번역)
  if (session.timer) clearTimeout(session.timer);
  session.timer = setTimeout(() => {
    flushSession(sessionId);
  }, BUFFER_FLUSH_MS);

  return res.json({
    sessionId,
    buffered: session.tokens,
    message: "Token received. Translation will trigger automatically."
  });
});

// 즉시 번역 요청 - 버퍼를 바로 비우고 번역
// POST /flush { sessionId: "..." }
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

// 세션 버퍼 초기화
// DELETE /session/:sessionId
app.delete("/session/:sessionId", (req, res) => {
  clearSession(req.params.sessionId);
  res.json({ ok: true, sessionId: req.params.sessionId });
});

// ── WebSocket ──────────────────────────────────────────────────────────────────
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// sessionId → WebSocket 클라이언트 맵
const wsClients = new Map();

// 타이머 만료 시 자동 번역 후 결과를 WebSocket으로 푸시
async function flushSession(sessionId) {
  const session = sessionBuffers.get(sessionId);
  if (!session || session.tokens.length === 0) return;

  const glosses = [...session.tokens];
  clearSession(sessionId);

  try {
    const result = await buildSentenceFromGlosses(glosses);
    const payload = JSON.stringify({
      type: "sentence",
      sessionId,
      glosses,
      sentence: result.sentence,
      cached: result.cached,
      model: defaultModel
    });

    // 해당 세션의 WebSocket 클라이언트에게 전송
    const ws = wsClients.get(sessionId);
    if (ws && ws.readyState === ws.OPEN) {
      ws.send(payload);
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

    // 세션 등록
    if (type === "join") {
      clientSessionId = sessionId;
      wsClients.set(sessionId, ws);
      ws.send(JSON.stringify({ type: "joined", sessionId }));
      return;
    }

    // 단어 한 개 수신
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

    // 즉시 번역
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

    // 단어 배열 일괄 번역 (기존 방식)
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
