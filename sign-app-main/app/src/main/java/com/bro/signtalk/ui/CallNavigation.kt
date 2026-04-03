// main/java/com/bro/signtalk/ui/CallNavigation.kt

package com.bro.signtalk.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.telecom.TelecomManager
import android.util.Log

object CallNavigation {
    private const val TAG = "SignTalk_Nav"

    // [필살기] 음성 통화 엔진 직접 가동!
    fun makeVoiceCall(context: Context, phoneNumber: String) {
        placeCarrierCall(context, phoneNumber, isVideo = false)
    }

    // [쌈뽕] 영상 통화 엔진 직접 가동!
    fun makeVideoCall(context: Context, phoneNumber: String) {
        placeCarrierCall(context, phoneNumber, isVideo = true)
    }

    // [쫀득] 문자 메시지 엔진!
    fun sendSms(context: Context, phoneNumber: String) {
        val intent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("smsto:$phoneNumber")
        }
        context.startActivity(intent)
    }

    // [팩폭] 어느 화면에서든 전@화를 터뜨리는 공통 엔진이다 유남생?!?
    private fun placeCarrierCall(context: Context, phoneNumber: String, isVideo: Boolean) {
        if (phoneNumber.isEmpty()) return
        val cleanNumber = phoneNumber.replace(Regex("[^0-9]"), "")
        val uri = Uri.parse("tel:$cleanNumber")

        try {
            val telecomManager = context.getSystemService(Context.TELECOM_SERVICE) as TelecomManager
            val extras = Bundle().apply {
                if (isVideo) {
                    putInt(TelecomManager.EXTRA_START_CALL_WITH_VIDEO_STATE, android.telecom.VideoProfile.STATE_BIDIRECTIONAL)
                }
            }
            // [핵심] 이제 SearchActivity에서도 전@화가 찰지게 터진다!
            telecomManager.placeCall(uri, extras)
        } catch (e: Exception) {
            Log.e(TAG, "전화 심부름 실패: ${e.message}")
        }
    }
}