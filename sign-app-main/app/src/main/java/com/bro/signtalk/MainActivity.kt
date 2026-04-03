package com.bro.signtalk

import android.app.role.RoleManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.telecom.TelecomManager
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.PopupMenu
import androidx.core.telecom.CallAttributesCompat
import androidx.core.telecom.CallsManager
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.bro.signtalk.ui.CallActivity
import com.bro.signtalk.ui.SettingsActivity
import com.google.android.material.bottomnavigation.BottomNavigationView
import kotlinx.coroutines.launch
import com.bro.signtalk.ui.DialpadFragment
import com.bro.signtalk.ui.recent.RecentCallsFragment
import com.bro.signtalk.ui.ContactsFragment
import android.provider.Settings

class MainActivity : AppCompatActivity() {
    private val TAG = "SignTalk_Main"
    private lateinit var callsManager: CallsManager

    // registerPhoneAccount랑 startManagedCall에서 쓰는 ID를 통일해라!
    // MainActivity.kt 에서 ID를 하나로 통일해라!
    private val HANDLE_ID = "SignTalk_Handle_ID"

    // MainActivity.kt - 이@게 진짜 정답이다 이말이야!

    // MainActivity.kt - 이게 진짜 갤럭시 전화 대체용 발신 엔진이다!

    // MainActivity.kt
    // MainActivity.kt

    // [팩폭] 이제 이 함수는 껍데기만 남겨라!
// 진짜 갤럭시 전화처럼 쓸 거면 시스템 엔진(SIM)이 대장이라서 등록할 필요가 없다 이말이야!
    private fun registerPhoneAccount(context: Context) {
        // 쫀득하게 로그만 남기고 아무것도 하지 마라!
        Log.d(TAG, "이제 가@짜 엔진 등록 따위 안 한다 이말이야! 유남생?!?")
    }

    // setupCoreTelecom도 깔@롱하게 정리해라!

    // [필살기] 이 함수 하나면 진짜 전화가 찰지게 터진다!
    // MainActivity.kt
    // MainActivity.kt 의 makeCarrierCall 함수를 싹 다 이걸로 교체해라!

    // [필살기] 영상/음성 다 받아주는 쌈뽕한 만능 엔진이다 이말이야!
    // [필살기] 영상/음성 다 받아주는 쌈뽕한 만능 엔진이다 이말이야!
    fun makeCarrierCall(phoneNumber: String, isVideo: Boolean = false) {
        if (phoneNumber.isEmpty()) {
            Toast.makeText(this, "번호 똑@바로 안 치냐?!? 팍@씨!", Toast.LENGTH_SHORT).show()
            return
        }

        // 숫자만 쫀득하게 남기기!
        val cleanNumber = phoneNumber.replace(Regex("[^0-9]"), "")
        val uri = Uri.parse("tel:$cleanNumber")

        try {
            if (checkSelfPermission(android.Manifest.permission.CALL_PHONE) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                val telecomManager = getSystemService(Context.TELECOM_SERVICE) as TelecomManager

                // [핵심] 시스템한테 "이건 영상통화다!"라고 편지(Extras)를 써서 같이 보내야 한다!
                val extras = Bundle().apply {
                    if (isVideo) {
                        putInt(TelecomManager.EXTRA_START_CALL_WITH_VIDEO_STATE, android.telecom.VideoProfile.STATE_BIDIRECTIONAL)
                    }
                }

                // 편지랑 같이 발신 신호탄 똬악!
                telecomManager.placeCall(uri, extras)
                Log.d(TAG, "통신사 망으로 진짜 전@화 심부름 보냈다 (영상모드: $isVideo) 유남생?!?")
            } else {
                // 권한 없으면 멱살 잡고 요청해라!
                requestPermissions(arrayOf(android.Manifest.permission.CALL_PHONE), 100)
            }
        } catch (e: Exception) {
            Log.e(TAG, "심부름 실패: ${e.message}")
        }
    }
    // [필살기] 에러 났던 RequestRole 대신 호환성 쩌는 StartActivityForResult로 원복!
    private val roleRequestLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        // 결과가 어떻든 일단 기강 잡으러 들어간다!
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val roleManager = getSystemService(RoleManager::class.java)
            if (roleManager.isRoleHeld(RoleManager.ROLE_DIALER)) {
                Log.d(TAG, "임명장 획득 완@료! 이제 내가 대장이다 ㅇㅇ.")
                setupCoreTelecom()
            } else {
                Toast.makeText(this, "기본 앱 설정을 해야 전@화를 받는다 브@로!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // 클래스 상단 (onCreate 위쪽)에 이게 있어야 에러 안 난다 유남생?!?


    // MainActivity.kt 수정안
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 1. 기본 앱 설정이랑 오버레이 권한을 여기서 먼저 굽@신굽@신 해라 ㅇㅇ
        checkOverlayPermission()

        // 2. [쌈뽕] 나머지 UI 리스너 및 네비게이션 설정 찰@지게 세팅!
        findViewById<View>(R.id.btn_main_more_menu)?.setOnClickListener { view ->
            val popup = PopupMenu(this, view)
            popup.menuInflater.inflate(R.menu.menu_main, popup.menu)
            popup.setOnMenuItemClickListener { item ->
                if (item.itemId == R.id.action_settings) {
                    startActivity(Intent(this, SettingsActivity::class.java))
                    true
                } else false
            }
            popup.show()
        }

        // MainActivity.kt 의 onCreate 내부
        // MainActivity.kt
        findViewById<View>(R.id.btn_main_search)?.setOnClickListener {
            // [필살기] 검색 화면으로 0.1초 만에 점프!
            val intent = Intent(this, com.bro.signtalk.ui.SearchActivity::class.java)
            startActivity(intent)
        }

        // 네비게이션 바 설정도 onCreate 뱃속에 얌전히 둬라 이말이야!
        val bottomNav = findViewById<BottomNavigationView>(R.id.bottom_navigation)
        if (savedInstanceState == null) loadFragment(DialpadFragment())

        bottomNav.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_dialpad -> loadFragment(DialpadFragment())
                R.id.nav_recent -> loadFragment(RecentCallsFragment())
                R.id.nav_contacts -> loadFragment(ContactsFragment())
                else -> false
            }
        }
    }

    // 3. [필살기] 화면이 확실히 뜰 때 권한 팝업을 멱@살 잡고 소환해라!
// 팩폭: onCreate 바깥으로 완전히 독립시켜야 에러가 안 난다 유남생?!?
    // MainActivity.kt 내부의 권한 확인 관련 함수들을 싹 다 이걸로 교체해라!

    override fun onResume() {
        super.onResume()
        Log.d(TAG, "앱 화면 복귀! 권한 기강 0.1초 만에 다시 잡는다 이말이야!")

        // 1. 다른 앱 위에 표시 권한부터 무조건 검문!
        checkOverlayPermission()

        // 2. 일반 권한 검문 (이게 통과되면 3번 대장 자리 검문으로 자동 연결된다!)
        checkAndRequestPermissions()
    }

    private fun checkAndRequestPermissions() {
        val permissions = arrayOf(
            android.Manifest.permission.READ_CONTACTS,
            android.Manifest.permission.CALL_PHONE,
            android.Manifest.permission.READ_PHONE_STATE,
            android.Manifest.permission.CAMERA,       // [쫀득] 영상통화 하려면 카메라도!
            android.Manifest.permission.RECORD_AUDIO  // [쫀득] 마이크도 여기서 한 번에!
        )

        val missingPermissions = permissions.filter {
            checkSelfPermission(it) != android.content.pm.PackageManager.PERMISSION_GRANTED
        }.toTypedArray()

        if (missingPermissions.isNotEmpty()) {
            // 권한이 하나라도 없으면 멱살 잡고 요청 팝업!
            requestPermissionsLauncher.launch(missingPermissions)
        } else {
            // [핵심/팩폭] 일반 권한이 다 있다면, 대장(기본 앱) 자리 안 뺏겼는지 확인해라!
            // 옛날엔 이 코드가 없어서 팝업이 안 떴던 거다 이말이야!
            checkDefaultDialerRole()
        }
    }

    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.entries.all { it.value }
        if (allGranted) {
            // 권한 방금 다 받았으면, 바로 이어서 대장 자리 팝업 똬악!
            checkDefaultDialerRole()
        } else {
            Toast.makeText(this, "권한 안 주면 전@화 못 한다 브@로! 설정 가서 켜라 팍@씨!", Toast.LENGTH_LONG).show()
        }
    }

    private fun checkDefaultDialerRole() {
        val telecomManager = getSystemService(Context.TELECOM_SERVICE) as TelecomManager

        // 이미 내가 굳건한 대장이면 얌전히 엔진만 가동해라!
        if (telecomManager.defaultDialerPackage == packageName) {
            Log.d(TAG, "임명장 확인 완@료! 내가 여전히 대장이다 ㅇㅇ.")
            setupCoreTelecom()
            return
        }

        // [쌈뽕] 대장 자리 뺏겼으면 묻지도 따지지도 않고 팝업 소환!
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val roleManager = getSystemService(Context.ROLE_SERVICE) as RoleManager
            if (!roleManager.isRoleHeld(RoleManager.ROLE_DIALER)) {
                Log.d(TAG, "대장 자리 뺏겼다! 다시 내놓으라고 RoleManager 팝업 쏜다!")
                val intent = roleManager.createRequestRoleIntent(RoleManager.ROLE_DIALER)
                roleRequestLauncher.launch(intent)
            }
        } else {
            // [팩폭] 안드로이드 10 미만 구형 기기용 기본 앱 설정 팝업이다!
            // 옛날 코드엔 빈 화면 띄우는 이상한 코드가 있었는데 이거 써야 진짜 팝업이 뜬다 유남생?!?
            val intent = Intent(TelecomManager.ACTION_CHANGE_DEFAULT_DIALER).apply {
                putExtra(TelecomManager.EXTRA_CHANGE_DEFAULT_DIALER_PACKAGE_NAME, packageName)
            }
            startActivityForResult(intent, 1004)
        }
    }

    private fun loadFragment(fragment: Fragment): Boolean {
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .commit()
        return true
    }

    private fun setupCoreTelecom() {
        callsManager = CallsManager(this)
        val capabilities = CallsManager.CAPABILITY_BASELINE or
                CallsManager.CAPABILITY_SUPPORTS_VIDEO_CALLING
        callsManager.registerAppWithTelecom(capabilities)

        // registerPhoneAccount(this) // 이것도 굳이 부를 필요 없다 이말이야!
        Log.d(TAG, "Jetpack Telecom SDK만 찰@지게 가동 완료! ㅇㅇ.")
    }



    // [필살기] 음성/영상 둘 다 찰지게 지원하는 발신 엔진!

    // MainActivity.kt에 추가해라 브로!
    private fun checkOverlayPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(this)) {
                // [쫀득] 권한 없으면 설정창으로 멱살 잡고 끌고 가기!
                val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName"))
                startActivity(intent)
                Toast.makeText(this, "수신 화면 띄우려면 '다른 앱 위에 표시' 권한 줘라 브@로!", Toast.LENGTH_LONG).show()
            }
        }
    }

    // MainActivity.kt 등에서 호출할 발신 코드다 이말이지!
    // MainActivity.kt - 진짜 전@화 터뜨리는 법!
    // [핵심] 프로젝트에서 startManagedCall 함수는 아예 지워버려라!
// 오직 이 'makeCarrierCall' 하나로 통일하는 거다 이말이야 ㅇㅇ.



    fun simulateIncomingCall() {
        val capabilities = CallAttributesCompat.SUPPORTS_SET_INACTIVE
        val attributes = CallAttributesCompat(
            "수@어 브@로",
            Uri.parse("tel:01012345678"),
            CallAttributesCompat.DIRECTION_INCOMING,
            CallAttributesCompat.CALL_TYPE_VIDEO_CALL,
            capabilities
        )

        lifecycleScope.launch {
            try {
                callsManager.addCall(
                    attributes,
                    onAnswer = { callType -> Log.d(TAG, "수락! 타입: $callType") },
                    onDisconnect = { cause -> Log.d(TAG, "종료: $cause") },
                    onSetActive = { Log.d(TAG, "활성!") },
                    onSetInactive = { Log.d(TAG, "대기!") }
                ) {
                    val intent = Intent(this@MainActivity, CallActivity::class.java).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    startActivity(intent)
                }
            } catch (e: Exception) {
                Log.e(TAG, "전화 추가 실패했다 이말이야: ${e.message}")
            }
        }
    }

    private fun launchCallUI() {
        val intent = Intent(this, CallActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        startActivity(intent)
        Log.d(TAG, "CallActivity 쌈@뽕하게 소환!")
    }

    fun getCurrentCall(): android.telecom.Call? {
        return com.bro.signtalk.service.SignCallService.currentCall
    }
}