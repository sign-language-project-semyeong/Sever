package com.bro.signtalk.ui

import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.media.Ringtone
import android.media.RingtoneManager
import android.net.Uri
import android.os.*
import android.provider.Settings
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import com.bro.signtalk.R

class SettingsSoundActivity : AppCompatActivity() {

    private lateinit var audioManager: AudioManager
    private lateinit var vibrator: Vibrator
    private var currentRingtoneUri: Uri? = null

    private var previewRingtone: Ringtone? = null

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            result.data?.data?.let { uri ->
                showPreviewDialog(uri)
            }
        }
    }

    private val ringtoneLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val uri = result.data?.getParcelableExtra<Uri>(RingtoneManager.EXTRA_RINGTONE_PICKED_URI)
            uri?.let {
                currentRingtoneUri = it
                applyRingtone(it)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings_sound)

        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        setupSoundSettings()
        setupVibrationSettings()
    }

    private fun setupSoundSettings() {
        val soundSwitch = findViewById<SwitchCompat>(R.id.switch_sound_enable)
        val ringtoneBtn = findViewById<Button>(R.id.btn_select_ringtone)
        val volumeSeekBar = findViewById<SeekBar>(R.id.seekbar_volume)
        val tvVolumeLevel = findViewById<TextView>(R.id.tv_volume_level)

        soundSwitch.isChecked = audioManager.ringerMode != AudioManager.RINGER_MODE_SILENT

        soundSwitch.setOnCheckedChangeListener { buttonView, isChecked ->
            // [쫀득] 다른 거 만지면 노래 바로 꺼라 팍씨!
            previewRingtone?.stop()

            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as android.app.NotificationManager

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !notificationManager.isNotificationPolicyAccessGranted) {
                buttonView.isChecked = !isChecked
                Toast.makeText(this, "무음 모드 바꾸려면 방해 금지 권한 내놔라 브@로!", Toast.LENGTH_LONG).show()
                startActivity(Intent(Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS))
                return@setOnCheckedChangeListener
            }

            try {
                audioManager.ringerMode = if (isChecked) AudioManager.RINGER_MODE_NORMAL else AudioManager.RINGER_MODE_SILENT
            } catch (e: SecurityException) {
                buttonView.isChecked = !isChecked
                Toast.makeText(this, "권한 에러 났다 팍씨!", Toast.LENGTH_SHORT).show()
            }
        }

        ringtoneBtn.setOnClickListener {
            // [쫀득] 여기도 버튼 누르는 순간 노래 컷!
            previewRingtone?.stop()

            val options = arrayOf("기본 벨소리 목록", "내 파일에서 직접 고르기")

            AlertDialog.Builder(this)
                .setTitle("어디서 가져올래 팍씨?!?")
                .setItems(options) { _, which ->
                    if (which == 0) {
                        val intent = Intent(RingtoneManager.ACTION_RINGTONE_PICKER).apply {
                            putExtra(RingtoneManager.EXTRA_RINGTONE_TYPE, RingtoneManager.TYPE_RINGTONE)
                            putExtra(RingtoneManager.EXTRA_RINGTONE_TITLE, "벨소리 골라라 브@로!")
                            putExtra(RingtoneManager.EXTRA_RINGTONE_SHOW_DEFAULT, true)
                            putExtra(RingtoneManager.EXTRA_RINGTONE_SHOW_SILENT, false)
                            putExtra(RingtoneManager.EXTRA_RINGTONE_EXISTING_URI, currentRingtoneUri)
                        }
                        ringtoneLauncher.launch(intent)
                    } else {
                        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                            type = "audio/*"
                        }
                        filePickerLauncher.launch(intent)
                    }
                }
                .show()
        }

        val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_RING)
        volumeSeekBar.max = maxVolume

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            volumeSeekBar.min = 1
        }

        val currentVol = audioManager.getStreamVolume(AudioManager.STREAM_RING).coerceAtLeast(1)
        volumeSeekBar.progress = currentVol
        tvVolumeLevel.text = "벨소리 볼륨: ${currentVol}단계"

        volumeSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                val safeVolume = if (progress == 0) 1 else progress
                tvVolumeLevel.text = "벨소리 볼륨: ${safeVolume}단계"

                if (fromUser) {
                    audioManager.setStreamVolume(AudioManager.STREAM_RING, safeVolume, 0)
                    if (progress == 0 && Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
                        sb?.progress = 1
                    }
                }
            }

            override fun onStartTrackingTouch(sb: SeekBar?) {
                // [팩폭] 바에 다시 손가락 올리는 순간 부르던 노래 입 틀어막아라!
                previewRingtone?.stop()
            }

            override fun onStopTrackingTouch(sb: SeekBar?) {
                playVolumePreview()
            }
        })
    }

    private fun setupVibrationSettings() {
        val prefs = getSharedPreferences("SignTalkSettings", Context.MODE_PRIVATE)
        val switchVib = findViewById<SwitchCompat>(R.id.switch_vibrate_enable)
        val seekVib = findViewById<SeekBar>(R.id.seekbar_vibration)
        val tvLevel = findViewById<TextView>(R.id.tv_vibration_level)

        val isEnabled = prefs.getBoolean("vibration_enabled", true)
        val intensity = prefs.getInt("vibration_intensity", 2)

        switchVib.isChecked = isEnabled
        seekVib.progress = intensity
        tvLevel.text = "진동 세기: ${intensity + 1}단계"

        seekVib.isEnabled = isEnabled
        tvLevel.alpha = if (isEnabled) 1.0f else 0.5f

        switchVib.setOnCheckedChangeListener { _, isChecked ->
            // [쌈뽕] 진동 건드려도 노래 얄짤없이 컷!
            previewRingtone?.stop()

            prefs.edit().putBoolean("vibration_enabled", isChecked).apply()
            seekVib.isEnabled = isChecked
            tvLevel.alpha = if (isChecked) 1.0f else 0.5f
            if (isChecked) playVibration(seekVib.progress)
        }

        seekVib.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                tvLevel.text = "진동 세기: ${progress + 1}단계"
                if (fromUser && switchVib.isChecked) {
                    prefs.edit().putInt("vibration_intensity", progress).apply()
                }
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {
                // [쌈뽕] 진동 바 건드려도 컷!
                previewRingtone?.stop()
            }
            override fun onStopTrackingTouch(sb: SeekBar?) {
                if (switchVib.isChecked) playVibration(sb?.progress ?: 2)
            }
        })
    }

    private fun showPreviewDialog(uri: Uri) {
        val ringtone = RingtoneManager.getRingtone(this, uri)
        ringtone?.play()

        AlertDialog.Builder(this)
            .setTitle("이 노래로 할 거냐?!?")
            .setMessage("지금 나오는 소리가 네가 설정한 볼륨이다 이말이야. 쌈뽕하면 적용 눌러라 유남생?!?")
            .setPositiveButton("ㅇㅇ 적용") { _, _ ->
                ringtone?.stop()
                currentRingtoneUri = uri
                applyRingtone(uri)
            }
            .setNegativeButton("ㄴㄴ 취소") { _, _ ->
                ringtone?.stop()
            }
            .setOnCancelListener {
                ringtone?.stop()
            }
            .show()
    }

    private fun applyRingtone(uri: Uri) {
        if (Settings.System.canWrite(this)) {
            RingtoneManager.setActualDefaultRingtoneUri(this, RingtoneManager.TYPE_RINGTONE, uri)
            Toast.makeText(this, "벨소리 변경 완료! ㅇㅇ.", Toast.LENGTH_SHORT).show()
        } else {
            val intent = Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS).apply {
                data = Uri.parse("package:$packageName")
            }
            startActivity(intent)
            Toast.makeText(this, "권한부터 줘라 브@로!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun playVolumePreview() {
        previewRingtone?.stop()
        val defaultUri = RingtoneManager.getActualDefaultRingtoneUri(this, RingtoneManager.TYPE_RINGTONE)
        previewRingtone = RingtoneManager.getRingtone(this, defaultUri)
        previewRingtone?.play()

        Handler(Looper.getMainLooper()).removeCallbacksAndMessages(null)
        Handler(Looper.getMainLooper()).postDelayed({
            previewRingtone?.stop()
        }, 30000)
    }

    private fun playVibration(progress: Int) {
        val amplitude = ((progress + 1) * 51).coerceAtMost(255)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(200, amplitude))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(200)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        previewRingtone?.stop()
    }
}