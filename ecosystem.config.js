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
        MODEL_NAME: "top30", // top30, top50, or demo
        NODE_SERVER_URL: "http://localhost:3000",
      },
    },
  ],
};
