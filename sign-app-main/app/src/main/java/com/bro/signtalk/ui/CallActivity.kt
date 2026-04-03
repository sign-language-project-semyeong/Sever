package com.bro.signtalk.ui

import android.widget.Chronometer
import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioManager
import android.net.Uri
import android.os.Bundle
import android.provider.ContactsContract
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.bro.signtalk.R
import com.bro.signtalk.service.SignCallService
import com.bro.signtalk.ui.contacts.VideoCallActivity

class CallActivity : AppCompatActivity() {

    private lateinit var audioManager: AudioManager
    private var isSpeakerOn = false
    private var isMuted = false

    // 수신기는 이거 하나면 충분하다 팍씨!
    private val callReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                "com.bro.signtalk.CALL_ENDED" -> {
                    Log.d("Call", "통화 종료 검거! 0.1초 만에 자폭한다!")
                    findViewById<Chronometer>(R.id.chronometer_call_time).stop()
                    finish()
                }
                "com.bro.signtalk.CALL_STARTED" -> {
                    Log.d("Call", "통화 연결 확인! 타이머 똬악 시작한다 이말이야! ㅇㅇ.")
                    val timer = findViewById<Chronometer>(R.id.chronometer_call_time)
                    timer.base = android.os.SystemClock.elapsedRealtime()
                    timer.visibility = View.VISIBLE
                    timer.start()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState) // 딱 한 번만 불러라!

        window.addFlags(
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                    WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                    WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
        )

        setContentView(R.layout.activity_call)

        // 두 가지 신호를 모두 잡는 찰진 필터!
        val filter = IntentFilter().apply {
            addAction("com.bro.signtalk.CALL_ENDED")
            addAction("com.bro.signtalk.CALL_STARTED")
        }

        ContextCompat.registerReceiver(
            this,
            callReceiver,
            filter,
            ContextCompat.RECEIVER_NOT_EXPORTED
        )

        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION

        setupUI()
    }

    private fun setupUI() {
        val number = intent.getStringExtra("receiver_phone") ?: ""
        val nameFromIntent = intent.getStringExtra("receiver_name")
        val displayName = if (!nameFromIntent.isNullOrEmpty() && nameFromIntent != "알 수 없음") {
            nameFromIntent
        } else {
            getContactName(number)
        }

        findViewById<TextView>(R.id.tv_call_name).text = displayName
        findViewById<TextView>(R.id.tv_call_number).text = number

        findViewById<View>(R.id.btn_speaker).setOnClickListener {
            isSpeakerOn = !isSpeakerOn
            it.isSelected = isSpeakerOn
            setSpeakerphone(isSpeakerOn)
        }

        findViewById<View>(R.id.btn_mute).setOnClickListener {
            isMuted = !isMuted
            audioManager.isMicrophoneMute = isMuted
            it.isSelected = isMuted
        }

        findViewById<View>(R.id.btn_keypad).setOnClickListener {
            val dialpad = findViewById<View>(R.id.layout_dialpad)
            if (dialpad.visibility == View.GONE) {
                dialpad.startAnimation(android.view.animation.AnimationUtils.loadAnimation(this, R.anim.slide_up))
                dialpad.visibility = View.VISIBLE
                it.isSelected = true
            } else {
                val slideDown = android.view.animation.AnimationUtils.loadAnimation(this, R.anim.slide_down)
                slideDown.setAnimationListener(object : android.view.animation.Animation.AnimationListener {
                    override fun onAnimationStart(animation: android.view.animation.Animation?) {}
                    override fun onAnimationEnd(animation: android.view.animation.Animation?) {
                        dialpad.visibility = View.GONE
                    }
                    override fun onAnimationRepeat(animation: android.view.animation.Animation?) {}
                })
                dialpad.startAnimation(slideDown)
                it.isSelected = false
            }
        }

        findViewById<View>(R.id.btn_video)?.setOnClickListener {
            SignCallService.currentCall?.let { call ->
                Log.d("Call", "음성에서 영상으로 변신! ㅇㅇ.")
                call.answer(android.telecom.VideoProfile.STATE_BIDIRECTIONAL)

                val intent = Intent(this, VideoCallActivity::class.java).apply {
                    putExtra("receiver_phone", number)
                    putExtra("receiver_name", displayName)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                }
                startActivity(intent)
                finish()
            }
        }

        setupKeypad()

        findViewById<View>(R.id.btn_hangup).setOnClickListener {
            SignCallService.currentCall?.disconnect()
            finish()
        }
    }

    private fun setupKeypad() {
        val buttonIds = arrayOf(R.id.btn_0, R.id.btn_1, R.id.btn_2, R.id.btn_3, R.id.btn_4, R.id.btn_5, R.id.btn_6, R.id.btn_7, R.id.btn_8, R.id.btn_9, R.id.btn_star, R.id.btn_pound)
        val digits = charArrayOf('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '#')
        buttonIds.forEachIndexed { index, id ->
            findViewById<View>(id)?.setOnClickListener { sendDtmf(digits[index]) }
        }
    }

    private fun sendDtmf(digit: Char) {
        SignCallService.currentCall?.let {
            it.playDtmfTone(digit)
            it.stopDtmfTone()
        }
    }

    private fun setSpeakerphone(on: Boolean) {
        SignCallService.instance?.setAudioRoute(
            if (on) android.telecom.CallAudioState.ROUTE_SPEAKER
            else android.telecom.CallAudioState.ROUTE_WIRED_OR_EARPIECE
        )
    }

    private fun getContactName(phoneNumber: String): String {
        val uri = Uri.withAppendedPath(ContactsContract.PhoneLookup.CONTENT_FILTER_URI, Uri.encode(phoneNumber))
        return contentResolver.query(uri, arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else phoneNumber
        } ?: phoneNumber
    }

    override fun onResume() {
        super.onResume()
        com.bro.signtalk.service.SignCallService.CallScreenTracker.isVisible = true
    }

    override fun onPause() {
        super.onPause()
        com.bro.signtalk.service.SignCallService.CallScreenTracker.isVisible = false
    }

    override fun onDestroy() {
        super.onDestroy()
        audioManager.mode = AudioManager.MODE_NORMAL
        // 해지할 때도 이거 하나면 끝이다 이말이야!
        try { unregisterReceiver(callReceiver) } catch (e: Exception) { }
    }
}