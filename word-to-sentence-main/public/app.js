const glossInput = document.getElementById("gloss-input");
const tokenPreview = document.getElementById("token-preview");
const submitButton = document.getElementById("submit-button");
const sentenceOutput = document.getElementById("sentence-output");
const jsonOutput = document.getElementById("json-output");
const errorMessage = document.getElementById("error-message");
const statusPill = document.getElementById("status-pill");
const exampleButtons = document.querySelectorAll(".example-button");
const sttButton = document.getElementById("stt-button");
const sttLabel = document.getElementById("stt-label");
const ttsButton = document.getElementById("tts-button");
const ttsLabel = document.getElementById("tts-label");

function parseGlosses(value) {
  return value
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function renderTokens() {
  const glosses = parseGlosses(glossInput.value);
  tokenPreview.innerHTML = "";

  if (glosses.length === 0) {
    const empty = document.createElement("div");
    empty.className = "token-empty";
    empty.textContent = "No glosses yet.";
    tokenPreview.appendChild(empty);
    return glosses;
  }

  glosses.forEach((gloss) => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    chip.textContent = gloss;
    tokenPreview.appendChild(chip);
  });

  return glosses;
}

function setStatus(type, text) {
  statusPill.className = `status-pill ${type}`;
  statusPill.textContent = text;
}

async function translateGlosses() {
  const glosses = renderTokens();

  if (glosses.length === 0) {
    errorMessage.hidden = false;
    errorMessage.textContent = "Please enter at least one gloss token.";
    setStatus("error", "Input needed");
    return;
  }

  submitButton.disabled = true;
  errorMessage.hidden = true;
  sentenceOutput.textContent = "Gemini is shaping a sentence...";
  jsonOutput.textContent = JSON.stringify({ glosses }, null, 2);
  setStatus("loading", "Generating");

  try {
    const response = await fetch("/translate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ glosses }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to generate a sentence.");
    }

    sentenceOutput.textContent = data.sentence;
    jsonOutput.textContent = JSON.stringify(data, null, 2);
    setStatus("success", "Done");
    ttsButton.disabled = false;
  } catch (error) {
    sentenceOutput.textContent = "The sentence could not be generated.";
    errorMessage.hidden = false;
    errorMessage.textContent = error.message;
    setStatus("error", "Error");
  } finally {
    submitButton.disabled = false;
  }
}

glossInput.addEventListener("input", () => {
  renderTokens();
  errorMessage.hidden = true;
  setStatus("idle", "Idle");
});

submitButton.addEventListener("click", translateGlosses);

glossInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    translateGlosses();
  }
});

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    glossInput.value = button.dataset.example || "";
    renderTokens();
    errorMessage.hidden = true;
    setStatus("idle", "Idle");
  });
});

renderTokens();

// ── STT (Speech-to-Text) ──────────────────────────────────────────────────────
const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  const recognition = new SpeechRecognition();
  recognition.lang = "ko-KR";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  let isListening = false;

  sttButton.addEventListener("click", () => {
    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  });

  recognition.addEventListener("start", () => {
    isListening = true;
    sttButton.classList.add("recording");
    sttLabel.textContent = "듣는 중...";
  });

  recognition.addEventListener("result", (event) => {
    const transcript = event.results[0][0].transcript;
    glossInput.value = transcript;
    renderTokens();
    errorMessage.hidden = true;
    setStatus("idle", "Idle");
  });

  recognition.addEventListener("end", () => {
    isListening = false;
    sttButton.classList.remove("recording");
    sttLabel.textContent = "음성 입력";
  });

  recognition.addEventListener("error", (event) => {
    isListening = false;
    sttButton.classList.remove("recording");
    sttLabel.textContent = "음성 입력";
    errorMessage.hidden = false;
    errorMessage.textContent = `음성 인식 오류: ${event.error}`;
  });
} else {
  sttButton.disabled = true;
  sttButton.title = "이 브라우저는 음성 인식을 지원하지 않습니다";
  sttLabel.textContent = "미지원";
}

// ── TTS (Text-to-Speech) ──────────────────────────────────────────────────────
if ("speechSynthesis" in window) {
  ttsButton.addEventListener("click", () => {
    const text = sentenceOutput.textContent.trim();
    if (
      !text ||
      text === "No sentence has been generated yet." ||
      text === "The sentence could not be generated."
    )
      return;

    if (speechSynthesis.speaking) {
      speechSynthesis.cancel();
      ttsButton.classList.remove("speaking");
      ttsLabel.textContent = "읽기";
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ko-KR";
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      ttsButton.classList.add("speaking");
      ttsLabel.textContent = "정지";
    };

    utterance.onend = () => {
      ttsButton.classList.remove("speaking");
      ttsLabel.textContent = "읽기";
    };

    utterance.onerror = () => {
      ttsButton.classList.remove("speaking");
      ttsLabel.textContent = "읽기";
    };

    speechSynthesis.speak(utterance);
  });
} else {
  ttsButton.disabled = true;
  ttsButton.title = "이 브라우저는 TTS를 지원하지 않습니다";
  ttsLabel.textContent = "미지원";
}
