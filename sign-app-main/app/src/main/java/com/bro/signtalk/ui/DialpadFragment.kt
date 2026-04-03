package com.bro.signtalk.ui

import android.app.AlertDialog // [핵심] 다이얼로그 띄우려면 이거 똬악 추가해야 한다 팍씨!
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.telecom.Call
import android.text.Editable
import android.text.InputFilter
import android.text.TextWatcher
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.bro.signtalk.MainActivity
import com.bro.signtalk.R

class DialpadFragment : Fragment() {

    private lateinit var audioManager: AudioManager
    private lateinit var input: EditText // [핵심] 다이얼로그에서도 접근하게 전역으로 빼라 이말이야!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        audioManager = requireContext().getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return inflater.inflate(R.layout.fragment_dialpad, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        input = view.findViewById(R.id.number_input)
        val videoBtn = view.findViewById<View>(R.id.btn_video_call)
        val deleteBtn = view.findViewById<View>(R.id.btn_delete)
        val addContactBtn = view.findViewById<ImageButton>(R.id.btn_add_contact)
        val callBtn = view.findViewById<View>(R.id.btn_call)

        input.filters = arrayOf(InputFilter.LengthFilter(15))
        input.showSoftInputOnFocus = false

        input.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                val hasText = s?.isNotEmpty() ?: false
                deleteBtn?.visibility = if (hasText) View.VISIBLE else View.INVISIBLE
                videoBtn?.visibility = if (hasText) View.VISIBLE else View.INVISIBLE
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })
        input.addTextChangedListener(android.telephony.PhoneNumberFormattingTextWatcher())

        // [수정] 리스트 대신 맵으로 바꿔서 번호 촥촥 뽑아 쓰게 갈아엎었다!
        val buttonMap = mapOf(
            R.id.btn_0 to "0", R.id.btn_1 to "1", R.id.btn_2 to "2",
            R.id.btn_3 to "3", R.id.btn_4 to "4", R.id.btn_5 to "5",
            R.id.btn_6 to "6", R.id.btn_7 to "7", R.id.btn_8 to "8",
            R.id.btn_9 to "9", R.id.btn_star to "*", R.id.btn_sharp to "#"
        )

        buttonMap.forEach { (viewId, digitStr) ->
            val btn = view.findViewById<Button>(viewId)

            // 1. [기본] 짧게 누르면 번호 입력!
            btn?.setOnClickListener {
                input.append(digitStr)
                val currentCall = (activity as? MainActivity)?.getCurrentCall()
                if (currentCall != null && currentCall.state == Call.STATE_ACTIVE) {
                    sendDtmfTone(currentCall, digitStr[0])
                }
            }

            // 2. [필살기] 숫자(0~9)를 길게 누르면 삼성 단축번호 엔진 발동!
            if (digitStr in "0".."9") {
                btn?.setOnLongClickListener {
                    // 입력창 숫자 + 방금 길게 누른 숫자 = 찐 단축번호 (ex: 1 누르고 2 길게 누르면 12번!)
                    val currentInput = input.text.toString()
                    val speedKeyStr = currentInput + digitStr
                    val speedKey = speedKeyStr.toIntOrNull()

                    if (speedKey != null && speedKey >= 0) {
                        val speedNumber = SpeedDialManager.getSpeedDialNumber(requireContext(), speedKey)

                        if (speedNumber != null) {
                            Log.d("SignTalk", "$speedKey 번 단축번호($speedNumber)로 쌈뽕하게 전화 건다!")
                            input.text.clear() // 팩폭: 통화 걸 거니까 입력창은 치워라!
                            (activity as? MainActivity)?.makeCarrierCall(speedNumber)
                        } else {
                            Log.d("SignTalk", "단축번호 $speedKey 번 비어있다 팍씨!")
                            showAssignSpeedDialDialog(speedKey)
                        }
                    }
                    true // 롱클릭 먹었으니까 짧은 클릭은 씹어라!
                }
            }
        }

        videoBtn?.setOnClickListener {
            val phoneNumber = input.text.toString()
            if (phoneNumber.isNotEmpty()) {
                CallNavigation.makeVideoCall(requireContext(), phoneNumber)
            } else {
                showToast("번호부터 치고 눌러라 브@로! 유남생?!?")
            }
        }

        callBtn?.setOnClickListener {
            val phoneNumber = input.text.toString()
            if (phoneNumber.isNotEmpty()) {
                (activity as? MainActivity)?.makeCarrierCall(phoneNumber)
            } else {
                showToast("전화번호가 없는데 어디다 걸어?!? 팍@씨!")
            }
        }

        deleteBtn?.setOnClickListener {
            val length = input.text.length
            if (length > 0) input.text.delete(length - 1, length)
        }

        deleteBtn?.setOnLongClickListener {
            input.text.clear()
            showToast("싹 다 성불시켰다 이말이야! ㅇㅇ.")
            true
        }

        addContactBtn?.setOnClickListener {
            val phoneNumber = input.text.toString()
            if (phoneNumber.isNotEmpty()) {
                val intent = Intent(requireContext(), AddContactActivity::class.java).apply {
                    putExtra("input_number", phoneNumber)
                }
                startActivity(intent)
            } else {
                showToast("번호부터 입력해라 브@로! 팍씨!")
            }
        }
    }

    // [쫀득] 단축번호 비어있을 때 띄우는 찰진 다이얼로그다 이말이야!
    // [필살기] 비어있는 단축번호 길게 눌렀을 때 실행되는 다이얼로그다!
    private fun showAssignSpeedDialDialog(key: Int) {
        AlertDialog.Builder(requireContext())
            .setTitle("단축번호 등록")
            .setMessage("단축번호 ${key}번에 지정된 연락처가 없다 이말이야. 지금 쌈뽕하게 등록하러 갈래 팍씨?!?")
            .setPositiveButton("ㅇㅇ 가자") { _, _ ->
                // [핵심] AddContactActivity가 아니라 SettingsSpeedDialActivity로 목적지 변경!
                val intent = Intent(requireContext(), SettingsSpeedDialActivity::class.java).apply {
                    // 선택했던 번호를 들고 가서 바로 세팅할 수 있게 배려해주는 센스!
                    putExtra("selected_speed_key", key)
                }
                startActivity(intent)
                input.text.clear() // 입력창은 깔끔하게 비워라 유남생?!?
            }
            .setNegativeButton("ㄴㄴ 취소", null)
            .show()
    }

    private fun sendDtmfTone(call: Call, digit: Char) {
        call.playDtmfTone(digit)
        Handler(Looper.getMainLooper()).postDelayed({
            call.stopDtmfTone()
        }, 200)
    }

    fun toggleSpeaker(isOn: Boolean) {
        audioManager.isSpeakerphoneOn = isOn
        showToast("스피커폰 ${if(isOn) "ON" else "OFF"} 했다 이말이야!")
    }

    object SpeedDialManager {
        fun getSpeedDialNumber(context: Context, key: Int): String? {
            val prefs = context.getSharedPreferences("SpeedDial", Context.MODE_PRIVATE)
            return prefs.getString(key.toString(), null)
        }

        fun saveSpeedDialNumber(context: Context, key: Int, phoneNumber: String) {
            val prefs = context.getSharedPreferences("SpeedDial", Context.MODE_PRIVATE)
            prefs.edit().putString(key.toString(), phoneNumber).apply()
        }
    }

    fun toggleMute(isMuted: Boolean) {
        audioManager.isMicrophoneMute = isMuted
        showToast("마이크 ${if(isMuted) "차단" else "해제"} 완료! ㅇㅇ.")
    }

    private fun showToast(message: String) {
        Toast.makeText(requireContext(), message, Toast.LENGTH_SHORT).show()
    }
}