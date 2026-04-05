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
import android.speech.tts.TextToSpeech
import java.util.Locale
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.ContactsContract
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
    private var tts: TextToSpeech? = null
    private var listenerWs: WebSocket? = null

    // ── 카메라 (CameraX) ──────────────────────────────────────────────────────
    private var cameraFacing = CameraSelector.LENS_FACING_FRONT
    private var cameraProvider: ProcessCameraProvider? = null
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var frameCounter = 0
    private val INFER_EVERY = 3   // 30fps 카메라에서 10fps로 추론

    // ── 네트워크 ──────────────────────────────────────────────────────────────
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()
    private var webSocket: WebSocket? = null
    private val sessionId = UUID.randomUUID().toString()

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

        // 상대방 문장 수신용 Android TTS 초기화
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.KOREAN
                tts?.setSpeechRate(0.85f)
                tts?.setAudioAttributes(
                    AudioAttributes.Builder()
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .build()
                )
            }
        }
        connectListenerWebSocket()
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
            SignCallService.instance?.setMuted(isMuted)
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
        // 통화 연결 후 올바른 순서로 오디오 설정 (에코 방지)
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioManager.isMicrophoneMute = false
        setSpeakerphone(true)
        runOnUiThread {
            findViewById<View>(R.id.layout_camera_loading).visibility = View.GONE
        }
        startCamera()
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
    // 리스너 WebSocket (상대방 문장 수신 → 내 폰 TTS 재생)
    // ─────────────────────────────────────────────────────────────────────────
    private fun connectListenerWebSocket() {
        val request = Request.Builder().url(NetworkConfig.WS_URL).build()
        listenerWs = httpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send("""{"type":"join","sessionId":"listener-$sessionId"}""")
                Log.d(TAG, "리스너 WebSocket 연결 완료")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    val originSession = json.optString("sessionId")
                    val isFromOther = originSession != sessionId

                    when (json.optString("type")) {
                        "sentence" -> {
                            val sentence = json.optString("sentence")
                            if (sentence.isNotEmpty() && isFromOther) {
                                runOnUiThread { addChatMessage(sentence, isMe = false) }
                            }
                        }
                        "audio" -> {
                            val audioB64 = json.optString("audio")
                            if (audioB64.isNotEmpty() && isFromOther) {
                                playReceivedAudio(audioB64)
                            }
                        }
                    }
                } catch (e: Exception) { }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "리스너 WS 실패: ${t.message}, 3초 후 재연결")
                mainHandler.postDelayed({ connectListenerWebSocket() }, 3000)
            }
        })
    }

    // 상대방 문장을 내 폰 스피커/이어폰으로 읽어주기
    private fun speakReceivedSentence(sentence: String) {
        tts?.speak(sentence, TextToSpeech.QUEUE_FLUSH, null, "incoming_tts")
        Log.d(TAG, "상대방 TTS 재생: $sentence")
    }

    // 서버에서 받은 base64 MP3를 STREAM_ALARM으로 재생 (통화 중에도 들림)
    private fun playReceivedAudio(base64Audio: String) {
        mainHandler.post {
            try {
                val bytes = Base64.decode(base64Audio, Base64.DEFAULT)
                val tempFile = File.createTempFile("recv_tts_", ".mp3", cacheDir)
                tempFile.writeBytes(bytes)

                mediaPlayer?.release()
                mediaPlayer = MediaPlayer().apply {
                    @Suppress("DEPRECATION")
                    setAudioStreamType(AudioManager.STREAM_ALARM)
                    setDataSource(tempFile.absolutePath)
                    prepare()
                    start()
                    setOnCompletionListener {
                        tempFile.delete()
                        release()
                    }
                }
                Log.d(TAG, "상대방 오디오 재생 (${bytes.size} bytes)")
            } catch (e: Exception) {
                Log.e(TAG, "상대방 오디오 재생 오류: ${e.message}")
            }
        }
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
            val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, false)
            val baos = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, 60, baos)
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
    // ─────────────────────────────────────────────────────────────────────────
    // TTS 오디오 재생 (서버에서 받은 base64 MP3)
    // STREAM_ALARM 으로 스피커 재생 → AEC 레퍼런스에서 제외
    // → 마이크가 TTS 소리를 집음해서 상대방에게 전달
    // ─────────────────────────────────────────────────────────────────────────
    private fun playTtsAudio(base64Audio: String) {
        mainHandler.post {
            try {
                val bytes = Base64.decode(base64Audio, Base64.DEFAULT)
                val tempFile = File.createTempFile("tts_out_", ".mp3", cacheDir)
                tempFile.writeBytes(bytes)

                isPlayingTts = true

                // 스피커폰 ON + 마이크 개방 → 스피커 소리를 마이크가 집음
                audioManager.isSpeakerphoneOn = true
                audioManager.isMicrophoneMute = false
                // TTS 볼륨 최대로
                audioManager.setStreamVolume(
                    AudioManager.STREAM_ALARM,
                    audioManager.getStreamMaxVolume(AudioManager.STREAM_ALARM),
                    0
                )

                mediaPlayer?.release()
                mediaPlayer = MediaPlayer().apply {
                    @Suppress("DEPRECATION")
                    setAudioStreamType(AudioManager.STREAM_ALARM)
                    setDataSource(tempFile.absolutePath)
                    prepare()
                    start()
                    setOnCompletionListener {
                        isPlayingTts = false
                        // 원래 스피커/마이크 상태 복구
                        audioManager.isSpeakerphoneOn = isSpeakerOn
                        audioManager.isMicrophoneMute = isMuted
                        // TTS 볼륨 원상복구
                        audioManager.setStreamVolume(
                            AudioManager.STREAM_ALARM,
                            audioManager.getStreamMaxVolume(AudioManager.STREAM_ALARM) / 2,
                            0
                        )
                        tempFile.delete()
                        release()
                        Log.d(TAG, "TTS 재생 완료")
                    }
                }
                Log.d(TAG, "TTS 재생 시작 (STREAM_ALARM, ${bytes.size} bytes)")
            } catch (e: Exception) {
                isPlayingTts = false
                Log.e(TAG, "TTS 재생 오류: ${e.message}")
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
            else android.telecom.CallAudioState.ROUTE_EARPIECE
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
        listenerWs?.close(1000, "Activity destroyed")

        mediaPlayer?.release()
        tts?.stop()
        tts?.shutdown()
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
            val tvMsg: TextView    = view.findViewById(R.id.tv_chat_message)
            val tvSender: TextView = view.findViewById(R.id.tv_sender)
            val viewBar: View      = view.findViewById(R.id.view_bar)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_chat_message, parent, false)
            return ViewHolder(view)
        }

        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            val msg = messages[position]
            holder.tvMsg.text = msg.text
            holder.tvMsg.background = null

            if (msg.isMe) {
                holder.tvSender.text = "나"
                holder.tvSender.setTextColor(0xFF6200EE.toInt())
                holder.viewBar.setBackgroundColor(0xFF6200EE.toInt())
            } else {
                holder.tvSender.text = "상대방"
                holder.tvSender.setTextColor(0xFF0288D1.toInt())
                holder.viewBar.setBackgroundColor(0xFF0288D1.toInt())
            }
        }

        override fun getItemCount() = messages.size
    }
}
