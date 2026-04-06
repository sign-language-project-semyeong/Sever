module.exports = {
  apps: [
    {
      name: "node-server",
      script: "server.js",
      cwd: "./word-to-sentence-main",
      env: {
        NODE_ENV: "production",
        PORT: 3000,
        PYTHON_SERVER_URL: "http://localhost:5000",
        AI_SERVER_URL: "http://localhost:5001",
      },
    },
    {
      name: "python-server",
      script: "app.py",
      cwd: "./sign-language-speech-main",
      interpreter: "py",
      wait_ready: true,
      listen_timeout: 10000,
    },
    {
      name: "ai-server",
      script: "ai_server.py",
      cwd: "./sign-language-ai-main/sign-language-ai-main",
      interpreter: "py",
      wait_ready: true,
      listen_timeout: 60000,
      env: {
        MODEL_NAME: "demo", // top30, top50, or demo
        CHECKPOINT_PATH:
          "C:/project/Sever/sign-language-ai-main/demo_gesture_2026-03-31_v1/models/best_gru_model.pt",
        NODE_SERVER_URL: "http://localhost:3000",
      },
    },
  ],
};
