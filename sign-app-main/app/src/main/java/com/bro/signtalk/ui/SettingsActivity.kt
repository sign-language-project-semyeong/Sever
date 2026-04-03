package com.bro.signtalk.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.bro.signtalk.R

class SettingsActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        // [필살기] 뒤로가기 버튼 활성화 (쌈뽕하게!)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "설정"

        // [쫀득] 벨소리(소리 및 진동) 설정 버튼 0.1초 만에 연결!
        findViewById<View>(R.id.btn_ringtone_settings).setOnClickListener {
            // [필살기] 진짜 SettingsSoundActivity로 점프한다 이말이야!
            val intent = Intent(this, SettingsSoundActivity::class.java)
            startActivity(intent)
            android.util.Log.d("Settings", "소리 및 진동 설정으로 쌈뽕하게 이동! ㅇㅇ.")
        }

        // [쌈뽕] 단축번호 설정 버튼 드디어 진짜 목적지 생겼다 이말이야!
        findViewById<View>(R.id.btn_speed_dial_settings).setOnClickListener {
            val intent = Intent(this, SettingsSpeedDialActivity::class.java)
            startActivity(intent)
            android.util.Log.d("Settings", "단축번호 관리로 찰지게 이동! ㅇㅇ.")
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        finish() // 뒤로가기 누르면 0.1초 만에 닫기!
        return true
    }
}