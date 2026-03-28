module.exports = {
  apps: [
    {
      name: "node-server",
      script: "./word_-to_sentence-main/server.js",
      cwd: "./word_-to_sentence-main",
      env: {
        NODE_ENV: "production",
        PORT: 3000,
      },
    },
    {
      name: "python-server",
      script: "app.py",
      cwd: "./sign-language-speech-main (1)/sign-language-speech-main",
      interpreter: "py",
    },
    {
      name: "ai-server",
      script: "ai_server.py",
      cwd: "./sign-language-ai-main/sign-language-ai-main",
      interpreter: "py",
    },
  ],
};
