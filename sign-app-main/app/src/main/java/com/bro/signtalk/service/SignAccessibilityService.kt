package com.bro.signtalk.service

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import android.util.Log

class SignAccessibilityService : AccessibilityService() {
    // [쫀득] 여기는 서비스 관련 로직만 똬악 있어야 한다!
    override fun onServiceConnected() {
        super.onServiceConnected()
        Log.d("SignTalk_Acc", "서비스 터널 개통 완료! ㅇㅇ.")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}
}

// [팩폭] 여기 밑에는 아무것도 없어야 한다 이말이야 유남생?!?
// SettingsActivity는 'ui' 패키지에 따로 있으니까 여기엔 필요 없다!