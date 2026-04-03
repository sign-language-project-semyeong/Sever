package com.bro.signtalk.ui

import android.content.ContentProviderOperation
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.ContactsContract
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.bro.signtalk.R
import java.io.ByteArrayOutputStream
import com.yalantis.ucrop.UCrop
import java.io.File

class AddContactActivity : AppCompatActivity() {

    private var selectedImageUri: Uri? = null
    private lateinit var ivPhoto: ImageView

    // 1. [필살기] 사진 고르는 배달 기사는 이거 하나면 충분하다!
    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        // [쫀득] 사진 골랐으면 딴짓 말고 바로 자르기(Crop) 드가라!
        uri?.let { startCrop(it) }
    }

    // 2. [필살기] 자르기 끝난 결과물 똬악 받아오는 전담 기사!
    private val cropResultLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val resultUri = UCrop.getOutput(result.data!!)
            resultUri?.let {
                selectedImageUri = it
                ivPhoto.setImageURI(it) // [찰진] 자른 사진 화면에 똬악!
            }
        }
    }

    // 3. [진짜 필살기] uCrop 세팅 (정사각형 + 원형 가이드)
    private fun startCrop(uri: Uri) {
        val destinationUri = Uri.fromFile(File(cacheDir, "temp_crop_${System.currentTimeMillis()}.jpg"))

        val options = UCrop.Options().apply {
            setCircleDimmedLayer(true) // [핵심] 원형 가이드라인!
            setShowCropGrid(false)
            setToolbarTitle("프로필 편집")
            // [쫀득] 테마 색깔도 쌈뽕하게 맞추고 싶으면 여기서 더 만져라!
        }

        val uCrop = UCrop.of(uri, destinationUri)
            .withAspectRatio(1f, 1f) // 무@조건 정사각형!
            .withMaxResultSize(480, 480) // 0.1초 만에 용량 방어!
            .withOptions(options)

        cropResultLauncher.launch(uCrop.getIntent(this))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_add_contact)

        ivPhoto = findViewById(R.id.iv_contact_photo)
        val etName = findViewById<EditText>(R.id.et_contact_name)
        val etPhone = findViewById<EditText>(R.id.et_contact_phone)
        val btnAdd = findViewById<Button>(R.id.btn_confirm_add)
        val btnCancel = findViewById<Button>(R.id.btn_cancel_add)

        // [팩폭] 사진 클릭 시 uCrop으로 이어지는 런처 실행!
        ivPhoto.setOnClickListener { imagePickerLauncher.launch("image/*") }

        val passedNumber = intent.getStringExtra("input_number") ?: ""
        etPhone.setText(passedNumber)

        val watcher = object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                val isReady = etName.text.isNotEmpty() && etPhone.text.isNotEmpty()
                btnAdd.isEnabled = isReady
                btnAdd.alpha = if (isReady) 1.0f else 0.5f
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        }

        etName.addTextChangedListener(watcher)
        etPhone.addTextChangedListener(watcher)
        btnCancel.setOnClickListener { finish() }

        btnAdd.setOnClickListener {
            val name = etName.text.toString()
            val phone = etPhone.text.toString()
            val existingContactId = findExistingContactId(phone)

            if (existingContactId != null) {
                showDuplicateDialog(existingContactId, name, phone, selectedImageUri)
            } else {
                // [쫀득] 여기서 저장할 때도 selectedImageUri(자른 사진)를 쓴다!
                saveContactDirectly(name, phone, selectedImageUri)
            }
        }
    }
    // --- [필살기] 모든 함수는 onCreate 밖에 독립적으로 박아라! ---

    private fun getOptimizedPhoto(uri: Uri): ByteArray? {
        return try {
            val originalBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri))
            } else {
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }

            val maxSize = 480
            var width = originalBitmap.width
            var height = originalBitmap.height
            val ratio = width.toFloat() / height.toFloat()

            if (ratio > 1) { width = maxSize; height = (maxSize / ratio).toInt() }
            else { height = maxSize; width = (maxSize * ratio).toInt() }

            val resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, width, height, true)
            val stream = ByteArrayOutputStream()
            // [팩폭] JPEG 80% 압축이 국룰이다 이말이야!
            resizedBitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
            val data = stream.toByteArray()

            if (originalBitmap != resizedBitmap) originalBitmap.recycle()
            resizedBitmap.recycle()
            data
        } catch (e: Exception) {
            Log.e("Photo", "최적화 실패: ${e.message}"); null
        }
    }

    private fun findExistingContactId(phoneNumber: String): String? {
        val uri = Uri.withAppendedPath(ContactsContract.PhoneLookup.CONTENT_FILTER_URI, Uri.encode(phoneNumber))
        val projection = arrayOf(ContactsContract.PhoneLookup.CONTACT_ID)
        return contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }
    }

    private fun showDuplicateDialog(contactId: String, name: String, phone: String, photo: Uri?) {
        android.app.AlertDialog.Builder(this)
            .setTitle("중@복 번호 발견했다 이말이야!")
            .setMessage("이미 있는 번호다 브@로! 수정할 거냐?!?")
            .setPositiveButton("수정") { _, _ -> updateContactDirectly(contactId, name, phone, photo) }
            .setNegativeButton("취소", null).show()
    }

    private fun updateContactDirectly(contactId: String, name: String, phone: String, photoUri: Uri?) {
        val ops = ArrayList<ContentProviderOperation>()

        ops.add(ContentProviderOperation.newUpdate(ContactsContract.Data.CONTENT_URI)
            .withSelection("${ContactsContract.Data.CONTACT_ID} = ? AND ${ContactsContract.Data.MIMETYPE} = ?",
                arrayOf(contactId, ContactsContract.CommonDataKinds.StructuredName.CONTENT_ITEM_TYPE))
            .withValue(ContactsContract.CommonDataKinds.StructuredName.DISPLAY_NAME, name).build())

        ops.add(ContentProviderOperation.newUpdate(ContactsContract.Data.CONTENT_URI)
            .withSelection("${ContactsContract.Data.CONTACT_ID} = ? AND ${ContactsContract.Data.MIMETYPE} = ?",
                arrayOf(contactId, ContactsContract.CommonDataKinds.Phone.CONTENT_ITEM_TYPE))
            .withValue(ContactsContract.CommonDataKinds.Phone.NUMBER, phone).build())

        photoUri?.let { uri ->
            getOptimizedPhoto(uri)?.let { data ->
                ops.add(ContentProviderOperation.newUpdate(ContactsContract.Data.CONTENT_URI)
                    .withSelection("${ContactsContract.Data.CONTACT_ID} = ? AND ${ContactsContract.Data.MIMETYPE} = ?",
                        arrayOf(contactId, ContactsContract.CommonDataKinds.Photo.CONTENT_ITEM_TYPE))
                    .withValue(ContactsContract.CommonDataKinds.Photo.PHOTO, data).build())
            }
        }

        try {
            contentResolver.applyBatch(ContactsContract.AUTHORITY, ops)
            Toast.makeText(this, "수정 완료했다 이말이야! ㅇㅇ.", Toast.LENGTH_SHORT).show()
            finish()
        } catch (e: Exception) { Log.e("Update", "에러: ${e.message}") }
    }

    private fun saveContactDirectly(name: String, phone: String, photoUri: Uri?) {
        val ops = ArrayList<ContentProviderOperation>()
        ops.add(ContentProviderOperation.newInsert(ContactsContract.RawContacts.CONTENT_URI)
            .withValue(ContactsContract.RawContacts.ACCOUNT_TYPE, null)
            .withValue(ContactsContract.RawContacts.ACCOUNT_NAME, null).build())

        ops.add(ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI)
            .withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
            .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.StructuredName.CONTENT_ITEM_TYPE)
            .withValue(ContactsContract.CommonDataKinds.StructuredName.DISPLAY_NAME, name).build())

        ops.add(ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI)
            .withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
            .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.Phone.CONTENT_ITEM_TYPE)
            .withValue(ContactsContract.CommonDataKinds.Phone.NUMBER, phone).build())

        photoUri?.let { uri ->
            getOptimizedPhoto(uri)?.let { data ->
                ops.add(ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI)
                    .withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                    .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.Photo.CONTENT_ITEM_TYPE)
                    .withValue(ContactsContract.CommonDataKinds.Photo.PHOTO, data).build())
            }
        }

        try {
            contentResolver.applyBatch(ContactsContract.AUTHORITY, ops)
            Toast.makeText(this, "저장 완료했다 이말이야! ㅇㅇ.", Toast.LENGTH_SHORT).show()
            finish()
        } catch (e: Exception) { Log.e("Save", "에러: ${e.message}") }
    }
}