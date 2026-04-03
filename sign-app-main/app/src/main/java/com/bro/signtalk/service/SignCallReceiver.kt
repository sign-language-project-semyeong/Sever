package com.bro.signtalk.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telecom.VideoProfile
import android.util.Log

class SignCallReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val call = SignCallService.currentCall ?: return

        when (intent.action) {
            "com.bro.signtalk.ACTION_ANSWER" -> {
                Log.d("SignCallReceiver", "알림창에서 찰@지게 전화 받았다 이말이야!")
                val videoState = call.details?.videoState ?: VideoProfile.STATE_AUDIO_ONLY
                val isVideo = videoState != VideoProfile.STATE_AUDIO_ONLY

                // [쫀득] 영상/음성 맞춰서 진짜 전화 받아라!
                call.answer(if (isVideo) VideoProfile.STATE_BIDIRECTIONAL else VideoProfile.STATE_AUDIO_ONLY)

                // [핵심/팩폭] 이거 때문에 앱이 죽은 거다! 이거 절대 쓰지 마라!
                // context.sendBroadcast(Intent(Intent.ACTION_CLOSE_SYSTEM_DIALOGS)) <--- 이거 날려버렸다 팍씨!

                // [쌈뽕] 대신 전화를 받았으니까 수신 화면이나 통화 화면으로 찰지게 보내버려라!
                // 시스템이 알아서 알림창을 뒤로 쑤셔 박아줄 거다 이말이야!
                val targetClass = if (isVideo) com.bro.signtalk.ui.contacts.VideoCallActivity::class.java else com.bro.signtalk.ui.CallActivity::class.java
                val callIntent = Intent(context, targetClass).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                    putExtra("receiver_phone", call.details.handle?.schemeSpecificPart)
                }
                context.startActivity(callIntent)
            }
            "com.bro.signtalk.HANGUP", "com.bro.signtalk.ACTION_REJECT" -> {
                Log.d("SignCallReceiver", "통화 쿨하게 씹었다 유남생?!?")
                call.disconnect()
            }
        }
    }
}