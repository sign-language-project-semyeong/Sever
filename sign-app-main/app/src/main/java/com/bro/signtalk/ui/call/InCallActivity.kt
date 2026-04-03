package com.bro.signtalk.ui.call

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

// 이 클래스는 사용되지 않음. VideoCallActivity(ui/contacts/)가 실제 수어통화 화면을 담당함.
class VideoCallActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        finish()
    }
}
