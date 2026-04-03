package com.bro.signtalk.service

import android.app.Notification
import android.app.NotificationChannel
import androidx.core.app.Person
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.media.Ringtone
import android.media.RingtoneManager
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.telecom.Call
import android.telecom.InCallService
import android.util.Log
import androidx.core.app.NotificationCompat
import com.bro.signtalk.R
import com.bro.signtalk.ui.CallActivity
import com.bro.signtalk.ui.IncomingCallActivity
import com.bro.signtalk.ui.contacts.VideoCallActivity

const val ACTION_HANGUP = "com.bro.signtalk.HANGUP"
const val ACTION_SPEAKER = "com.bro.signtalk.SPEAKER"
const val ACTION_MUTE = "com.bro.signtalk.MUTE"

class SignCallService : InCallService() {

    private val TAG = "SignTalk_Service"
    private var ringtone: Ringtone? = null
    private var vibrator: Vibrator? = null

    companion object {
        var currentCall: Call? = null
        var instance: SignCallService? = null
        const val NOTIFICATION_ID = 1004
        // [핵심] v3로 바꿔서 기존 쓰레기 채널 버리고 새로 파라!
        const val CHANNEL_ID = "call_service_channel_v3"
    }

    // [쌈뽕] 화면 보고 있는지 추적하는 놈!
    object CallScreenTracker {
        var isVisible = false
    }

    private fun createCallNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // [팩폭] 소리랑 진동 끄는 옵션 다 날려버려야 팝업(Heads-up)이 찰지게 뜬다!
            val channel = NotificationChannel(CHANNEL_ID, "수어 통화 서비스 (중요)", NotificationManager.IMPORTANCE_HIGH).apply {
                description = "브로의 쌈뽕한 통화 알림"
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAlerts()
        instance = null
    }

    override fun onCallAdded(call: Call) {
        super.onCallAdded(call)
        currentCall = call
        call.registerCallback(callCallback)

        // [팩폭] supportsVideo 쓰레기 코드 날려버리고 진짜 상태만 체크해라!
        val videoState = call.details?.videoState ?: android.telecom.VideoProfile.STATE_AUDIO_ONLY
        val isVideoCall = videoState != android.telecom.VideoProfile.STATE_AUDIO_ONLY

        if (call.state == Call.STATE_RINGING) {
            startAlerts()

            val targetActivity = if (isVideoCall) {
                com.bro.signtalk.ui.IncomingVideoCallActivity::class.java
            } else {
                IncomingCallActivity::class.java
            }

            val intent = Intent(this, targetActivity).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                putExtra("receiver_phone", call.details.handle?.schemeSpecificPart)
            }
            startActivity(intent)

            showCallNotification(call.details.handle?.schemeSpecificPart ?: "", isIncoming = true, isConnected = false)
        } else {
            val targetActivity = if (isVideoCall) {
                VideoCallActivity::class.java
            } else {
                VideoCallActivity::class.java
            }

            val intent = Intent(this, targetActivity).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                putExtra("receiver_phone", call.details.handle?.schemeSpecificPart)
            }
            startActivity(intent)
        }
    }

    private val callCallback = object : Call.Callback() {
        override fun onStateChanged(call: Call, state: Int) {
            val phoneNumber = call.details.handle?.schemeSpecificPart ?: "알 수 없음"

            if (state == Call.STATE_ACTIVE || state == Call.STATE_DISCONNECTED) {
                stopAlerts()
            }

            if (state == Call.STATE_ACTIVE) {
                showCallNotification(phoneNumber, isIncoming = false, isConnected = true)
                sendBroadcast(Intent("com.bro.signtalk.CALL_STARTED").apply {
                    setPackage(packageName)
                })
                val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            }

            if (state == Call.STATE_DISCONNECTED) {
                cleanUpCallSession(call)
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        createCallNotificationChannel()

        val initialNotification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_call)
            .setContentTitle("SignTalk 서비스")
            .setContentText("전화 대기 중이다 이말이야 ㅇㅇ.")
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, initialNotification, ServiceInfo.FOREGROUND_SERVICE_TYPE_PHONE_CALL)
        } else {
            startForeground(NOTIFICATION_ID, initialNotification)
        }

        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }

    private fun showCallNotification(phoneNumber: String, isIncoming: Boolean, isConnected: Boolean) {
        val displayName = getContactName(phoneNumber)

        // [핵심] 여기서 영상 통화 오작동 원인 제거했다!
        val videoState = currentCall?.details?.videoState ?: android.telecom.VideoProfile.STATE_AUDIO_ONLY
        val isVideoCall = videoState != android.telecom.VideoProfile.STATE_AUDIO_ONLY

        val targetActivity = when {
            isIncoming && !isConnected && isVideoCall -> com.bro.signtalk.ui.IncomingVideoCallActivity::class.java
            isIncoming && !isConnected -> IncomingCallActivity::class.java
            isConnected && isVideoCall -> VideoCallActivity::class.java
            else -> VideoCallActivity::class.java
        }

        val activityIntent = Intent(this, targetActivity).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            putExtra("receiver_phone", phoneNumber)
            putExtra("receiver_name", displayName)
        }

        val pendingActivity = PendingIntent.getActivity(this, 0, activityIntent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)

        // 거절 및 끊기 인텐트
        val hangupIntent = Intent(this, SignCallReceiver::class.java).apply { action = ACTION_HANGUP }
        val pendingHangup = PendingIntent.getBroadcast(this, 1, hangupIntent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)

        // [쫀득] 받기 인텐트 부활! 진짜로 전화를 받게 신호를 쏴라!
        val answerIntent = Intent(this, SignCallReceiver::class.java).apply { action = "com.bro.signtalk.ACTION_ANSWER" }
        val pendingAnswer = PendingIntent.getBroadcast(this, 2, answerIntent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)

        val person = Person.Builder().setName(displayName).build()

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_call)
            // [쌈뽕] 화면 안에 있으면 LOW로 낮춰서 팝업 안 뜨게 하고, 바탕화면이면 MAX로 팍 띄워라!
            .setPriority(if (CallScreenTracker.isVisible) NotificationCompat.PRIORITY_LOW else NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setFullScreenIntent(pendingActivity, true)
            .setOngoing(true)
            .setAutoCancel(false)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)

        if (!isConnected) {
            builder.setContentTitle(if (isIncoming) "전화 왔다 이말이야!" else "전화 거는 중이다 이말이야!")
                .setContentText("$displayName 브로와 연결 중...")
                .setUsesChronometer(false)
                .setStyle(if (isIncoming) {
                    // [핵심] pendingActivity가 아니라 pendingAnswer를 넣어야 진짜로 받아진다 유남생?!?
                    NotificationCompat.CallStyle.forIncomingCall(person, pendingHangup, pendingAnswer)
                } else {
                    NotificationCompat.CallStyle.forOngoingCall(person, pendingHangup)
                })
        } else {
            builder.setContentTitle("$displayName 브로와 통화 중!")
                .setContentText(if (isVideoCall) "쌈뽕한 영상 통화 중..." else "쫀득한 음성 대화 중...")
                .setUsesChronometer(true)
                .setWhen(System.currentTimeMillis())
                .setStyle(NotificationCompat.CallStyle.forOngoingCall(person, pendingHangup))
        }

        val notification = builder.build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_PHONE_CALL)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun getContactName(phoneNumber: String): String {
        if (phoneNumber.isEmpty()) return "알 수 없음"
        val uri = android.net.Uri.withAppendedPath(android.provider.ContactsContract.PhoneLookup.CONTENT_FILTER_URI, android.net.Uri.encode(phoneNumber))
        val projection = arrayOf(android.provider.ContactsContract.PhoneLookup.DISPLAY_NAME)

        return contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else phoneNumber
        } ?: phoneNumber
    }

    private fun stopCallNotification() {
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    override fun onCallRemoved(call: Call) {
        super.onCallRemoved(call)
        cleanUpCallSession(call)
    }

    private fun cleanUpCallSession(call: Call) {
        stopCallNotification()
        stopAlerts()

        if (currentCall == call) currentCall = null
        call.unregisterCallback(callCallback)

        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audioManager.mode = AudioManager.MODE_NORMAL

        // [필살기] 통화 끝났다고 동네방네 방송해야 화면들이 알아서 닫히고 키패드로 돌아간다!
        sendBroadcast(Intent("com.bro.signtalk.CALL_ENDED").apply { setPackage(packageName) })
    }

    private fun startAlerts() {
        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        if (audioManager.ringerMode != AudioManager.RINGER_MODE_SILENT) {
            // 벨소리 로직 (기존 코드 유지)
            try {
                val uri = RingtoneManager.getActualDefaultRingtoneUri(this, RingtoneManager.TYPE_RINGTONE)
                ringtone = RingtoneManager.getRingtone(this, uri)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) ringtone?.isLooping = true
                ringtone?.play()
            } catch (e: Exception) { }

            // [핵심] 설정 창고(SignTalkSettings) 털기 드간다!
            val prefs = getSharedPreferences("SignTalkSettings", Context.MODE_PRIVATE)
            val isVibrationEnabled = prefs.getBoolean("vibration_enabled", true)
            val intensityLevel = prefs.getInt("vibration_intensity", 2) // 기본 2단계

            if (isVibrationEnabled) {
                // [쌈뽕] 0~4 단계를 50~250 진동 세기로 찰지게 매핑!
                val amplitude = ((intensityLevel + 1) * 50).coerceAtMost(255)
                val pattern = longArrayOf(0, 1000, 1000) // 진동 주기
                val amplitudes = intArrayOf(0, amplitude, 0) // 세기 설정

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    // [필살기] 진짜 세기가 적용되는 웨이브폼 엔진 가동!
                    vibrator?.vibrate(VibrationEffect.createWaveform(pattern, amplitudes, 0))
                } else {
                    vibrator?.vibrate(pattern, 0)
                }
            }
        }
    }

    private fun stopAlerts() {
        ringtone?.stop()
        ringtone = null
        vibrator?.cancel()
    }
}