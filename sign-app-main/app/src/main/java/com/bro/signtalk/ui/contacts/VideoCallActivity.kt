package com.bro.signtalk.ui.contacts

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.ContactsContract
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Base64
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.bro.signtalk.NetworkConfig
import com.bro.signtalk.R
import com.bro.signtalk.service.SignCallService
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class VideoCallActivity : AppCompatActivity() {

    private val TAG = "SignCall"

    // ── 오디오 ────────────────────────────────────────────────────────────────
    private lateinit var audioManager: AudioManager
    private var isMuted = false
    private var isSpeakerOn = true
    private var mediaPlayer: MediaPlayer? = null
    private var isPlayingTts = false

    // ── 카메라 (CameraX) ──────────────────────────────────────────────────────
    private var cameraFacing = CameraSelector.LENS_FACING_FRONT
    private var cameraProvider: ProcessCameraProvider? = null
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var frameCounter = 0
    private val INFER_EVERY = 6   // 30fps 카메라에서 5fps로 추론

    // ── 네트워크 ──────────────────────────────────────────────────────────────
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()
    private var webSocket: WebSocket? = null
    private val sessionId = UUID.randomUUID().toString()

    // ── STT ───────────────────────────────────────────────────────────────────
    private var speechRecognizer: SpeechRecognizer? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    // ── UI ────────────────────────────────────────────────────────────────────
    private lateinit var rvChat: RecyclerView
    private lateinit var chipGroup: ChipGroup
    private lateinit var btnGenerate: View
    private lateinit var chatAdapter: ChatAdapter
    private val chatMessages = mutableListOf<ChatMessage>()
    private val wordTokens = mutableListOf<String>()
    private var callConnectedFlag = false

    // ── 브로드캐스트 리시버 ───────────────────────────────────────────────────
    private val callReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                "com.bro.signtalk.CALL_ENDED" -> {
                    Log.d(TAG, "통화 종료")
                    finish()
                }
                "com.bro.signtalk.CALL_STARTED" -> {
                    Log.d(TAG, "통화 연결됨 → 수어 모드 시작")
                    runOnUiThread { onCallConnected() }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        window.addFlags(
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
            WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
        )
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_call)

        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        setSpeakerphone(true)

        // 발신 번호 표시
        val number = intent.getStringExtra("receiver_phone") ?: ""
        findViewById<TextView>(R.id.tv_call_name).text = getContactName(number)
        findViewById<TextView>(R.id.tv_call_number).text = number

        // UI 바인딩
        rvChat = findViewById(R.id.rv_chat)
        chipGroup = findViewById(R.id.chip_group_words)
        btnGenerate = findViewById(R.id.btn_generate_sentence)

        setupRecyclerView()
        setupButtons()
        connectWebSocket()

        val filter = IntentFilter().apply {
            addAction("com.bro.signtalk.CALL_ENDED")
            addAction("com.bro.signtalk.CALL_STARTED")
        }
        ContextCompat.registerReceiver(this, callReceiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)

        // 이미 통화 중인 상태로 액티비티가 열리는 경우 (알림에서 복귀 등)
        if (SignCallService.currentCall?.state == android.telecom.Call.STATE_ACTIVE) {
            onCallConnected()
        }

        // [폴백] 5초 안에 통화 연결 브로드캐스트가 안 오면 강제로 카메라 시작
        mainHandler.postDelayed({
            if (!callConnectedFlag) {
                Log.w(TAG, "폴백: CALL_STARTED 미수신 → 카메라 강제 시작")
                onCallConnected()
            }
        }, 5000)

        // 서버 연결 상태 확인
        checkServerHealth()
    }

    // ── RecyclerView 세팅 ─────────────────────────────────────────────────────
    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter(chatMessages)
        rvChat.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        rvChat.adapter = chatAdapter
    }

    // ── 버튼 세팅 ────────────────────────────────────────────────────────────
    private fun setupButtons() {
        // 카메라 전후면 전환
        findViewById<ImageButton>(R.id.btn_switch_camera).setOnClickListener {
            cameraFacing = if (cameraFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
            startCamera()
        }

        // 음소거
        val btnMute = findViewById<ImageButton>(R.id.btn_mute)
        btnMute.alpha = 0.4f
        btnMute.setOnClickListener {
            isMuted = !isMuted
            audioManager.isMicrophoneMute = isMuted
            (it as ImageButton).alpha = if (isMuted) 1.0f else 0.4f
        }

        // 스피커폰
        val btnSpeaker = findViewById<ImageButton>(R.id.btn_speaker)
        btnSpeaker.alpha = 1.0f
        btnSpeaker.setOnClickListener {
            isSpeakerOn = !isSpeakerOn
            setSpeakerphone(isSpeakerOn)
            (it as ImageButton).alpha = if (isSpeakerOn) 1.0f else 0.4f
        }

        // 종료
        findViewById<View>(R.id.btn_hangup).setOnClickListener {
            SignCallService.currentCall?.disconnect()
            finish()
        }

        // 문장 생성 (수동 flush)
        btnGenerate.setOnClickListener {
            if (wordTokens.isNotEmpty()) {
                webSocket?.send("""{"type":"flush"}""")
                Log.d(TAG, "수동 문장 생성 요청: $wordTokens")
            }
        }
    }

    // ── 통화 연결 시 호출 ─────────────────────────────────────────────────────
    private fun onCallConnected() {
        if (callConnectedFlag) return   // 중복 호출 방지
        callConnectedFlag = true
        runOnUiThread {
            findViewById<View>(R.id.layout_camera_loading).visibility = View.GONE
        }
        startCamera()
        startStt()
        Log.d(TAG, "수어 통화 화면 활성화 완료")
    }

    // ── 서버 헬스체크 (앱 시작 시 서버 상태 확인) ─────────────────────────────
    private fun checkServerHealth() {
        val req = Request.Builder()
            .url("${NetworkConfig.BASE_URL}/health")
            .build()
        httpClient.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    android.widget.Toast.makeText(
                        this@VideoCallActivity,
                        "⚠ 서버 연결 실패: ${NetworkConfig.BASE_URL}\n서버가 실행 중인지 확인하세요",
                        android.widget.Toast.LENGTH_LONG
                    ).show()
                }
                Log.e(TAG, "서버 헬스체크 실패: ${e.message}")
            }
            override fun onResponse(call: Call, response: Response) {
                Log.d(TAG, "서버 연결 OK: ${response.code}")
                response.close()
            }
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // WebSocket (Node.js 서버 연결)
    // ─────────────────────────────────────────────────────────────────────────
    private fun connectWebSocket() {
        val request = Request.Builder().url(NetworkConfig.WS_URL).build()
        webSocket = httpClient.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send("""{"type":"join","sessionId":"$sessionId"}""")
                Log.d(TAG, "WebSocket 연결 완료, 세션: $sessionId")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    when (json.optString("type")) {
                        "joined" -> Log.d(TAG, "세션 합류 확인")

                        // 수어 단어 버퍼 업데이트 (서버에서 실시간으로 옴)
                        "buffered" -> {
                            val arr = json.getJSONArray("tokens")
                            val tokens = (0 until arr.length()).map { arr.getString(it) }
                            runOnUiThread { syncWordChips(tokens) }
                        }

                        // Gemini가 만든 최종 문장
                        "sentence" -> {
                            val sentence = json.optString("sentence")
                            if (sentence.isNotEmpty()) {
                                runOnUiThread { addChatMessage(sentence, isMe = true) }
                                Log.d(TAG, "문장 수신: $sentence")
                            }
                        }

                        // TTS 오디오 (base64 mp3)
                        "audio" -> {
                            val audioB64 = json.optString("audio")
                            if (audioB64.isNotEmpty()) {
                                playTtsAudio(audioB64)
                            }
                        }

                        "error" -> Log.e(TAG, "서버 에러: ${json.optString("error")}")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "WebSocket 메시지 파싱 오류: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket 연결 실패: ${t.message}")
                // 3초 후 재연결 시도
                mainHandler.postDelayed({ connectWebSocket() }, 3000)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket 종료: $reason")
            }
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CameraX + 수어 추론 (AI 서버 POST /infer)
    // ─────────────────────────────────────────────────────────────────────────
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        val previewView = findViewById<PreviewView>(R.id.pv_local_camera)

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
            frameCounter++
            if (frameCounter % INFER_EVERY == 0) {
                sendFrameToAI(imageProxy)
            } else {
                imageProxy.close()
            }
        }

        val selector = CameraSelector.Builder()
            .requireLensFacing(cameraFacing)
            .build()

        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, imageAnalysis)
            Log.d(TAG, "카메라 바인딩 완료")
        } catch (e: Exception) {
            Log.e(TAG, "카메라 바인딩 실패: ${e.message}")
        }
    }

    private fun sendFrameToAI(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxy.toBitmap()
            imageProxy.close()

            // 320x240으로 축소 → JPEG 70% 압축 (전송 속도 최적화)
            val scaled = Bitmap.createScaledBitmap(bitmap, 320, 240, false)
            val baos = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, 70, baos)
            val b64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

            val bodyJson = """{"sessionId":"$sessionId","frameData":"$b64","fps":5}"""
            val body = bodyJson.toRequestBody("application/json".toMediaType())

            val req = Request.Builder()
                .url("${NetworkConfig.BASE_URL}/infer")
                .post(body)
                .build()

            httpClient.newCall(req).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    // 네트워크 오류는 무시 (다음 프레임에서 재시도)
                }

                override fun onResponse(call: Call, response: Response) {
                    val text = response.body?.string() ?: return
                    try {
                        val json = JSONObject(text)
                        val token = json.optString("committedToken", "")
                        // committedToken이 실제 단어인 경우에만 칩 추가 + 서버 버퍼에 전송
                        if (token.isNotEmpty() && token != "null") {
                            Log.d(TAG, "수어 단어 감지: $token")
                            // 서버 버퍼에 토큰 등록 → Gemini 자동 문장 생성 트리거
                            webSocket?.send("""{"type":"token","token":"$token"}""")
                            runOnUiThread { addWordChip(token) }
                        }
                    } catch (e: Exception) { }
                }
            })
        } catch (e: Exception) {
            try { imageProxy.close() } catch (_: Exception) {}
            Log.e(TAG, "프레임 처리 오류: ${e.message}")
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STT (상대방 목소리 → 채팅 텍스트)
    // ─────────────────────────────────────────────────────────────────────────
    private fun startStt() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Log.w(TAG, "이 기기에서 STT를 지원하지 않음")
            return
        }
        speechRecognizer?.destroy()
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onEvent(eventType: Int, params: Bundle?) {}

            override fun onResults(results: Bundle?) {
                val text = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                    ?: ""
                if (text.isNotEmpty()) {
                    Log.d(TAG, "STT 결과: $text")
                    runOnUiThread { addChatMessage(text, isMe = false) }
                }
                // TTS 재생 중이 아니면 계속 듣기
                if (!isPlayingTts) listenStt()
            }

            override fun onPartialResults(partial: Bundle?) {
                // 부분 결과는 표시 안 함 (노이즈 많음)
            }

            override fun onError(error: Int) {
                // NO_MATCH(7), CLIENT(5) 같은 일반 오류는 그냥 재시작
                if (!isPlayingTts) {
                    mainHandler.postDelayed({ listenStt() }, 300)
                }
            }
        })
        listenStt()
    }

    private fun listenStt() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 2000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 300L)
        }
        try {
            speechRecognizer?.startListening(intent)
        } catch (e: Exception) {
            Log.e(TAG, "STT 시작 오류: ${e.message}")
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TTS 오디오 재생 (서버에서 받은 base64 MP3)
    // MP3 → PCM 디코딩 후 AudioTrack(STREAM_VOICE_CALL)으로 전화 업링크에 직접 주입
    // → 상대방에게 TTS 음성이 들림
    // ─────────────────────────────────────────────────────────────────────────
    private fun playTtsAudio(base64Audio: String) {
        isPlayingTts = true
        speechRecognizer?.stopListening()
        audioManager.isMicrophoneMute = false

        cameraExecutor.execute {
            var audioTrack: AudioTrack? = null
            var extractor: MediaExtractor? = null
            var codec: MediaCodec? = null
            val tempFile = File.createTempFile("tts_out_", ".mp3", cacheDir)
            try {
                val bytes = Base64.decode(base64Audio, Base64.DEFAULT)
                tempFile.writeBytes(bytes)

                // ── MP3 → PCM16 디코딩 ───────────────────────────────────────
                extractor = MediaExtractor().also { it.setDataSource(tempFile.absolutePath) }
                var audioTrackIdx = -1
                var format: MediaFormat? = null
                for (i in 0 until extractor.trackCount) {
                    val f = extractor.getTrackFormat(i)
                    if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                        audioTrackIdx = i; format = f; break
                    }
                }
                if (audioTrackIdx < 0 || format == null) {
                    Log.e(TAG, "TTS: 오디오 트랙 없음"); return@execute
                }
                extractor.selectTrack(audioTrackIdx)

                val sampleRate  = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                val channels    = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                val channelMask = if (channels == 1) AudioFormat.CHANNEL_OUT_MONO
                                  else AudioFormat.CHANNEL_OUT_STEREO

                val minBuf = AudioTrack.getMinBufferSize(
                    sampleRate, channelMask, AudioFormat.ENCODING_PCM_16BIT)

                // STREAM_VOICE_CALL 로 전화 업링크에 직접 주입
                @Suppress("DEPRECATION")
                audioTrack = AudioTrack(
                    AudioManager.STREAM_VOICE_CALL,
                    sampleRate, channelMask,
                    AudioFormat.ENCODING_PCM_16BIT,
                    minBuf * 4,
                    AudioTrack.MODE_STREAM
                )
                audioTrack.play()

                codec = MediaCodec.createDecoderByType(
                    format.getString(MediaFormat.KEY_MIME)!!)
                codec.configure(format, null, null, 0)
                codec.start()

                val info    = MediaCodec.BufferInfo()
                var sawEOS  = false
                val timeout = 10_000L

                while (!sawEOS) {
                    // 입력 버퍼에 압축 데이터 공급
                    val inIdx = codec.dequeueInputBuffer(timeout)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val n   = extractor.readSampleData(buf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawEOS = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, n,
                                extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                    // 출력 버퍼에서 PCM 꺼내 AudioTrack으로 전송
                    val outIdx = codec.dequeueOutputBuffer(info, timeout)
                    if (outIdx >= 0) {
                        val pcm = codec.getOutputBuffer(outIdx)!!
                        val pcmBytes = ByteArray(info.size)
                        pcm.get(pcmBytes)
                        audioTrack.write(pcmBytes, 0, pcmBytes.size)
                        codec.releaseOutputBuffer(outIdx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0)
                            sawEOS = true
                    }
                }
                audioTrack.stop()
                Log.d(TAG, "TTS 재생 완료 (업링크 주입)")
            } catch (e: Exception) {
                Log.e(TAG, "TTS 재생 오류: ${e.message}")
            } finally {
                try { codec?.stop(); codec?.release() } catch (_: Exception) {}
                try { extractor?.release() } catch (_: Exception) {}
                try { audioTrack?.release() } catch (_: Exception) {}
                tempFile.delete()
                isPlayingTts = false
                audioManager.isMicrophoneMute = isMuted
                mainHandler.post { listenStt() }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 단어 칩 관리
    // ─────────────────────────────────────────────────────────────────────────

    // AI 서버에서 직접 감지된 단어를 칩으로 추가
    private fun addWordChip(token: String) {
        wordTokens.add(token)
        val chip = Chip(this).apply {
            text = token
            isCloseIconVisible = true
            setOnCloseIconClickListener {
                wordTokens.remove(token)
                chipGroup.removeView(this)
                btnGenerate.visibility = if (wordTokens.isEmpty()) View.GONE else View.VISIBLE
            }
        }
        chipGroup.addView(chip)
        btnGenerate.visibility = View.VISIBLE
    }

    // WebSocket "buffered" 이벤트로 서버 상태와 동기화
    private fun syncWordChips(tokens: List<String>) {
        wordTokens.clear()
        wordTokens.addAll(tokens)
        chipGroup.removeAllViews()
        tokens.forEach { token ->
            val chip = Chip(this).apply {
                text = token
                isCloseIconVisible = false
            }
            chipGroup.addView(chip)
        }
        btnGenerate.visibility = if (tokens.isEmpty()) View.GONE else View.VISIBLE
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 채팅 메시지 추가
    // ─────────────────────────────────────────────────────────────────────────
    private fun addChatMessage(text: String, isMe: Boolean) {
        chatMessages.add(ChatMessage(text, isMe))
        chatAdapter.notifyItemInserted(chatMessages.size - 1)
        rvChat.scrollToPosition(chatMessages.size - 1)

        // 내가 보낸 문장이면 칩 초기화
        if (isMe) {
            wordTokens.clear()
            chipGroup.removeAllViews()
            btnGenerate.visibility = View.GONE
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 유틸
    // ─────────────────────────────────────────────────────────────────────────
    private fun setSpeakerphone(on: Boolean) {
        audioManager.isSpeakerphoneOn = on
        SignCallService.instance?.setAudioRoute(
            if (on) android.telecom.CallAudioState.ROUTE_SPEAKER
            else android.telecom.CallAudioState.ROUTE_WIRED_OR_EARPIECE
        )
    }

    private fun getContactName(phoneNumber: String): String {
        if (phoneNumber.isEmpty()) return "알 수 없음"
        val uri = Uri.withAppendedPath(
            ContactsContract.PhoneLookup.CONTENT_FILTER_URI,
            Uri.encode(phoneNumber)
        )
        return contentResolver.query(
            uri, arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME), null, null, null
        )?.use {
            if (it.moveToFirst()) it.getString(0) else phoneNumber
        } ?: phoneNumber
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 라이프사이클
    // ─────────────────────────────────────────────────────────────────────────
    override fun onResume() {
        super.onResume()
        SignCallService.CallScreenTracker.isVisible = true
    }

    override fun onPause() {
        super.onPause()
        SignCallService.CallScreenTracker.isVisible = false
    }

    override fun onDestroy() {
        super.onDestroy()
        try { unregisterReceiver(callReceiver) } catch (_: Exception) {}

        // 세션 삭제 요청
        webSocket?.send("""{"type":"close","sessionId":"$sessionId"}""")
        webSocket?.close(1000, "Activity destroyed")

        speechRecognizer?.destroy()
        mediaPlayer?.release()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()

        Log.d(TAG, "VideoCallActivity 정리 완료")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 채팅 RecyclerView 어댑터 (내 말풍선 / 상대방 말풍선)
    // ─────────────────────────────────────────────────────────────────────────
    data class ChatMessage(val text: String, val isMe: Boolean)

    inner class ChatAdapter(private val messages: List<ChatMessage>) :
        RecyclerView.Adapter<ChatAdapter.ViewHolder>() {

        inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
            val tvMsg: TextView = view.findViewById(R.id.tv_chat_message)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_chat_message, parent, false)
            return ViewHolder(view)
        }

        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            val msg = messages[position]
            holder.tvMsg.text = msg.text

            val lp = holder.tvMsg.layoutParams as? FrameLayout.LayoutParams
                ?: FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )

            if (msg.isMe) {
                // 내 말풍선: 오른쪽 정렬, 핑크 배경
                lp.gravity = Gravity.END
                lp.marginStart = 80
                lp.marginEnd = 0
                holder.tvMsg.setBackgroundResource(R.drawable.bg_chat_bubble_mine)
                holder.tvMsg.setTextColor(0xFF1A1A1A.toInt())
            } else {
                // 상대방 말풍선: 왼쪽 정렬, 흰색 배경
                lp.gravity = Gravity.START
                lp.marginStart = 0
                lp.marginEnd = 80
                holder.tvMsg.setBackgroundResource(R.drawable.bg_chat_bubble_theirs)
                holder.tvMsg.setTextColor(0xFF1A1A1A.toInt())
            }
            holder.tvMsg.layoutParams = lp
        }

        override fun getItemCount() = messages.size
    }
}
