package com.bro.signtalk.ui.contacts

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognizerIntent
import android.util.Log
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.bro.signtalk.NetworkConfig
import com.bro.signtalk.R
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * 수어 영상통화 화면
 *
 * 흐름:
 *   [카메라 프레임] → POST /infer (AI 서버)
 *     → 단어 감지 → Node.js /token 자동 전송
 *     → Gemini 번역 + Python TTS
 *     → WebSocket으로 { type:"audio", audio:"base64..." } 수신
 *     → 앱에서 오디오 재생 + 문장 표시
 *
 *   [마이크 버튼] → 안드로이드 음성 인식 → 텍스트 화면 표시
 */
class VideoCallActivity : AppCompatActivity() {

    // ── UI ────────────────────────────────────────────────────────────────────
    private lateinit var previewView: PreviewView
    private lateinit var tvSentence: TextView
    private lateinit var tvStatus: TextView
    private lateinit var btnMic: ImageButton
    private lateinit var btnHangup: ImageButton

    // ── 카메라 ─────────────────────────────────────────────────────────────────
    private lateinit var cameraExecutor: ExecutorService
    private var lastFrameSentMs = 0L
    private val FRAME_INTERVAL_MS = 300L  // 약 3fps

    // ── 네트워크 ────────────────────────────────────────────────────────────────
    private val sessionId = UUID.randomUUID().toString()
    private var webSocket: WebSocket? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    // ── 오디오 재생 ─────────────────────────────────────────────────────────────
    private var mediaPlayer: MediaPlayer? = null

    // ── 음성 인식 (STT) ─────────────────────────────────────────────────────────
    private val speechLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val text = result.data
                ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
                ?.firstOrNull()
            if (!text.isNullOrEmpty()) {
                tvSentence.text = "🎙 $text"
            }
        }
    }

    // ── 권한 요청 ───────────────────────────────────────────────────────────────
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions[Manifest.permission.CAMERA] == true) {
            startCamera()
        } else {
            Toast.makeText(this, "카메라 권한이 필요합니다", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    // ── 생명주기 ────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_call)

        previewView = findViewById(R.id.preview_view)
        tvSentence  = findViewById(R.id.tv_sentence)
        tvStatus    = findViewById(R.id.tv_status)
        btnMic      = findViewById(R.id.btn_mic)
        btnHangup   = findViewById(R.id.btn_hangup)

        cameraExecutor = Executors.newSingleThreadExecutor()

        connectWebSocket()
        checkPermissionsAndStartCamera()

        btnMic.setOnClickListener { startSpeechRecognition() }
        btnHangup.setOnClickListener { finish() }
    }

    override fun onDestroy() {
        super.onDestroy()
        webSocket?.close(1000, "Activity destroyed")
        cameraExecutor.shutdown()
        mediaPlayer?.release()
        httpClient.dispatcher.executorService.shutdown()
    }

    // ── WebSocket 연결 ──────────────────────────────────────────────────────────
    private fun connectWebSocket() {
        val request = Request.Builder().url(NetworkConfig.WS_URL).build()

        webSocket = httpClient.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send(JSONObject().apply {
                    put("type", "join")
                    put("sessionId", sessionId)
                }.toString())
                updateStatus("🟢 연결됨")
                Log.d(TAG, "WebSocket 연결, sessionId=$sessionId")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    when (json.getString("type")) {

                        "joined" -> Log.d(TAG, "세션 등록 완료")

                        // 번역된 문장 수신
                        "sentence" -> {
                            val sentence = json.getString("sentence")
                            mainHandler.post { tvSentence.text = "🤟 $sentence" }
                        }

                        // 번역 + TTS 오디오 수신
                        "audio" -> {
                            val audioBase64 = json.getString("audio")
                            val sentence = json.optString("sentence", "")
                            mainHandler.post {
                                if (sentence.isNotEmpty()) tvSentence.text = "🤟 $sentence"
                                playAudio(audioBase64)
                            }
                        }

                        "buffered" -> Log.d(TAG, "버퍼: ${json.optJSONArray("tokens")}")

                        "error" -> Log.e(TAG, "서버 오류: ${json.optString("error")}")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "메시지 파싱 오류: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket 끊김: ${t.message}")
                updateStatus("🔴 연결 끊김 - 재연결 중...")
                mainHandler.postDelayed({ connectWebSocket() }, 3000)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                updateStatus("연결 종료")
            }
        })
    }

    // ── 카메라 권한 + 시작 ─────────────────────────────────────────────────────
    private fun checkPermissionsAndStartCamera() {
        val needed = buildList {
            if (!hasPermission(Manifest.permission.CAMERA))       add(Manifest.permission.CAMERA)
            if (!hasPermission(Manifest.permission.RECORD_AUDIO)) add(Manifest.permission.RECORD_AUDIO)
        }
        if (needed.isEmpty()) startCamera()
        else permissionLauncher.launch(needed.toTypedArray())
    }

    private fun hasPermission(permission: String) =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    // ── CameraX 시작 ────────────────────────────────────────────────────────────
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val now = System.currentTimeMillis()
                if (now - lastFrameSentMs >= FRAME_INTERVAL_MS) {
                    lastFrameSentMs = now
                    sendFrameToAI(imageProxy)
                } else {
                    imageProxy.close()
                }
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )
                Log.d(TAG, "카메라 시작됨")
            } catch (e: Exception) {
                Log.e(TAG, "카메라 바인딩 실패: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── 프레임 → AI 서버 전송 (/infer) ─────────────────────────────────────────
    private fun sendFrameToAI(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxy.toBitmap()
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 50, outputStream)
            val base64Frame = android.util.Base64.encodeToString(
                outputStream.toByteArray(), android.util.Base64.NO_WRAP
            )

            val body = JSONObject().apply {
                put("sessionId", sessionId)
                put("frameData", base64Frame)
                put("fps", 3.0)
            }.toString()

            val request = Request.Builder()
                .url("${NetworkConfig.BASE_URL}/infer")
                .post(body.toRequestBody("application/json".toMediaType()))
                .build()

            httpClient.newCall(request).enqueue(object : Callback {
                override fun onResponse(call: Call, response: Response) { response.close() }
                override fun onFailure(call: Call, e: java.io.IOException) {
                    Log.w(TAG, "Infer 요청 실패: ${e.message}")
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "프레임 전송 오류: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    // ── TTS 오디오 재생 ─────────────────────────────────────────────────────────
    private fun playAudio(base64Audio: String) {
        try {
            val audioBytes = android.util.Base64.decode(base64Audio, android.util.Base64.DEFAULT)
            val tempFile = File(cacheDir, "tts_audio.mp3")
            FileOutputStream(tempFile).use { it.write(audioBytes) }

            mediaPlayer?.release()
            mediaPlayer = MediaPlayer().apply {
                setDataSource(tempFile.absolutePath)
                prepare()
                start()
                setOnCompletionListener { release() }
            }
        } catch (e: Exception) {
            Log.e(TAG, "오디오 재생 오류: ${e.message}")
        }
    }

    // ── 음성 인식 (STT) ─────────────────────────────────────────────────────────
    private fun startSpeechRecognition() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_PROMPT, "말씀하세요...")
        }
        try {
            speechLauncher.launch(intent)
        } catch (e: Exception) {
            Toast.makeText(this, "음성 인식을 지원하지 않는 기기입니다", Toast.LENGTH_SHORT).show()
        }
    }

    // ── 헬퍼 ────────────────────────────────────────────────────────────────────
    private fun updateStatus(status: String) {
        mainHandler.post { tvStatus.text = status }
    }

    companion object {
        private const val TAG = "VideoCallActivity"
    }
}
