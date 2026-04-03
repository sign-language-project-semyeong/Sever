package com.bro.signtalk.ui

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.bro.signtalk.R
import com.bro.signtalk.service.SignCallService
import com.bro.signtalk.ui.contacts.VideoCallActivity

class IncomingVideoCallActivity : AppCompatActivity() {

    private var initialX = 0f
    private var phoneNumber: String = ""

    private val callEndedReceiver = object : android.content.BroadcastReceiver() {
        override fun onReceive(context: android.content.Context?, intent: android.content.Intent?) {
            if (intent?.action == "com.bro.signtalk.CALL_ENDED") {
                finish() // [쫀득] 상대가 끊으면 미련 없이 화면 닫고 키패드로 찰지게 돌아가라 이말이야!
            }
        }
    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(
            android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
                    android.view.WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                    android.view.WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                    android.view.WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
        )
        setContentView(R.layout.activity_incoming_video_call)

        phoneNumber = intent.getStringExtra("receiver_phone") ?: ""
        val displayName = if (phoneNumber.isNotEmpty()) getContactName(phoneNumber) else "알 수 없는 브@로"

        findViewById<TextView>(R.id.tv_incoming_name).text = displayName
        findViewById<TextView>(R.id.tv_incoming_number)?.text = phoneNumber

        val handle = findViewById<ImageView>(R.id.iv_swipe_handle)
        setupSwipeListener(handle)

        // [쫀득] 리시버 달아서 중간에 끊기면 바로 닫히게 기강 잡아라!
        val filter = android.content.IntentFilter("com.bro.signtalk.CALL_ENDED")
        androidx.core.content.ContextCompat.registerReceiver(this, callEndedReceiver, filter, androidx.core.content.ContextCompat.RECEIVER_NOT_EXPORTED)
    }

    private fun answerCall() {
        // [핵심] 영상통화니까 여기서 BIDIRECTIONAL로 대답해야 안 튕긴다 이말이야!
        SignCallService.currentCall?.answer(android.telecom.VideoProfile.STATE_BIDIRECTIONAL)
        val intent = Intent(this, VideoCallActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            putExtra("receiver_phone", phoneNumber)
        }
        startActivity(intent)
        finish()
    }

    private fun setupSwipeListener(handle: View) {
        handle.setOnTouchListener { view, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> { initialX = event.rawX; true }
                MotionEvent.ACTION_MOVE -> {
                    val diff = event.rawX - initialX
                    if (Math.abs(diff) < 400) view.translationX = diff
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val finalDiff = event.rawX - initialX
                    when {
                        finalDiff > 200 -> answerCall()
                        finalDiff < -200 -> {
                            SignCallService.currentCall?.disconnect()
                            finish()
                        }
                        else -> view.animate().translationX(0f).setDuration(200).start()
                    }
                    true
                }
                else -> false
            }
        }
    }

    private fun getContactName(phoneNumber: String): String {
        val uri = android.net.Uri.withAppendedPath(
            android.provider.ContactsContract.PhoneLookup.CONTENT_FILTER_URI,
            android.net.Uri.encode(phoneNumber)
        )
        val projection = arrayOf(android.provider.ContactsContract.PhoneLookup.DISPLAY_NAME)
        return contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else phoneNumber
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
        try { unregisterReceiver(callEndedReceiver) } catch (e: Exception) { }
    }
}