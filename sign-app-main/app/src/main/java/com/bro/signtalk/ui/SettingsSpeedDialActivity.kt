package com.bro.signtalk.ui

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.ContactsContract
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.bro.signtalk.R
import com.bro.signtalk.data.model.Contact

class SettingsSpeedDialActivity : AppCompatActivity() {

    private lateinit var btnSelectKey: Button
    private lateinit var etSearch: EditText
    private lateinit var listLayout: LinearLayout

    private var selectedKey: Int? = null
    private val allContacts = mutableListOf<Contact>()
    private var searchPopup: ListPopupWindow? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings_speed_dial)

        btnSelectKey = findViewById(R.id.btn_select_speed_key)
        etSearch = findViewById(R.id.et_speed_search)
        listLayout = findViewById(R.id.layout_speed_dial_list)

        btnSelectKey.setOnClickListener { showSpeedKeyPicker(it) }
        loadContactsFromSystem()

        etSearch.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                val query = s?.toString()?.trim() ?: ""
                if (query.isNotEmpty()) performSearch(query) else searchPopup?.dismiss()
            }
            override fun beforeTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {}
            override fun onTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {}
        })

        findViewById<Button>(R.id.btn_add_speed_dial).setOnClickListener {
            val key = selectedKey
            val numberStr = etSearch.text.toString().trim()

            if (key != null && numberStr.isNotEmpty()) {
                DialpadFragment.SpeedDialManager.saveSpeedDialNumber(this, key, numberStr)
                Toast.makeText(this, "${key}번에 쌈뽕하게 저장 완료!", Toast.LENGTH_SHORT).show()

                selectedKey = null
                btnSelectKey.text = "숫자선택▼"
                etSearch.text.clear()
                refreshSpeedDialList()
            } else {
                Toast.makeText(this, "번호 선택하고 연락처도 입력해라 팍씨!", Toast.LENGTH_SHORT).show()
            }
        }

        intent.getIntExtra("selected_speed_key", -1).let {
            if (it != -1) {
                selectedKey = it
                btnSelectKey.text = "${it}번 ▼"
            }
        }
        refreshSpeedDialList()
    }

    // [핵심] 주소록 로딩할 때 묻혀있던 프사(PHOTO_URI)까지 싹 다 캐온다 이말이야!
    private fun loadContactsFromSystem() {
        allContacts.clear()
        val cursor = contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            null, null, null,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " ASC"
        )
        cursor?.use {
            val nameIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
            val numberIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
            val idIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.CONTACT_ID)
            // [쫀득] 프사 경로 위치 확보!
            val photoUriIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.PHOTO_URI)

            while (it.moveToNext()) {
                val name = it.getString(nameIndex) ?: ""
                val number = it.getString(numberIndex) ?: ""
                val id = it.getString(idIndex) ?: ""
                val photoUri = it.getString(photoUriIndex) // 프사 캐오기!

                if (allContacts.none { c -> c.phoneNumber == number }) {
                    // [쌈뽕] Contact 객체에 프사 경로 찰지게 넣어준다!
                    allContacts.add(Contact(id, name, number, photoUri))
                }
            }
        }
    }

    // [필살기] 하이픈 무시 검색 & 연락처 UI 팝업 띄우기!
    private fun performSearch(query: String) {
        val cleanQuery = query.replace("-", "")

        val filtered = allContacts.filter {
            val cleanPhoneNumber = it.phoneNumber.replace("-", "")
            it.name.contains(query, ignoreCase = true) || cleanPhoneNumber.contains(cleanQuery)
        }.sortedWith(compareBy<Contact> {
            val index = it.name.indexOf(query, ignoreCase = true)
            if (index == -1) 999 else index
        }.thenBy { it.name })

        if (filtered.isEmpty()) {
            searchPopup?.dismiss()
            return
        }

        if (searchPopup == null) {
            searchPopup = ListPopupWindow(this).apply {
                anchorView = etSearch
                // [쫀득] 무식하게 큰 MATCH_PARENT 버리고 230dp로 다이어트 똬악!
                width = dpToPx(230)

                // [쌈뽕] 약간 오른쪽으로 밀어서 찰진 좌우 여백 확보!
                horizontalOffset = dpToPx(-20)

                height = dpToPx(300) // 대략 4~5개 보이는 높이 유지
            }
        }

        // 진짜 연락처 UI(item_contact.xml)를 어댑터에 똬악!
        val adapter = ContactSearchAdapter(this, filtered)
        searchPopup?.setAdapter(adapter)

        searchPopup?.setOnItemClickListener { _, _, position, _ ->
            val selectedContact = filtered[position]
            // 선택하면 하이픈 지운 생 번호로 입력창에 박아라!
            val rawNum = selectedContact.phoneNumber.replace("-", "")
            etSearch.setText(rawNum)
            etSearch.setSelection(rawNum.length)
            searchPopup?.dismiss()
        }

        searchPopup?.show()
    }

    inner class ContactSearchAdapter(context: Context, private val items: List<Contact>) : ArrayAdapter<Contact>(context, 0, items) {
        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = convertView ?: LayoutInflater.from(context).inflate(R.layout.item_contact, parent, false)
            val contact = items[position]

            val tvName = view.findViewById<TextView>(R.id.tv_contact_name)
            val tvNumber = view.findViewById<TextView>(R.id.tv_contact_number)
            val tvIcon = view.findViewById<TextView>(R.id.tv_profile_icon)
            val ivProfile = view.findViewById<ImageView>(R.id.iv_profile_image)
            val expandLayout = view.findViewById<LinearLayout>(R.id.layout_expand)
            val blockIcon = view.findViewById<ImageView>(R.id.iv_block_status)

            tvName.text = contact.name
            tvNumber.text = contact.phoneNumber
            expandLayout.visibility = View.GONE
            blockIcon.visibility = View.GONE

            // [필살기] 프사가 있으면 프사를 띄우고, 없으면 첫 글자 띄워라 팍씨!
            if (!contact.photoUri.isNullOrEmpty()) {
                ivProfile.setImageURI(Uri.parse(contact.photoUri))
                ivProfile.visibility = View.VISIBLE
                tvIcon.visibility = View.GONE
            } else if (contact.name.isNotEmpty()) {
                tvIcon.text = contact.name.first().toString()
                tvIcon.visibility = View.VISIBLE
                ivProfile.visibility = View.GONE
            }

            return view
        }
    }

    private fun showSpeedKeyPicker(anchor: View) {
        val prefs = getSharedPreferences("SpeedDial", Context.MODE_PRIVATE)
        val usedKeys = prefs.all.keys.mapNotNull { it.toIntOrNull() }.toSet()
        val availableKeys = (1..99).filter { it !in usedKeys }.map { it.toString() }

        ListPopupWindow(this).apply {
            setAdapter(ArrayAdapter(this@SettingsSpeedDialActivity, android.R.layout.simple_list_item_1, availableKeys))
            anchorView = anchor
            width = dpToPx(120)
            height = dpToPx(250)
            isModal = true
            setOnItemClickListener { _, _, position, _ ->
                val keyStr = availableKeys[position]
                selectedKey = keyStr.toInt()
                btnSelectKey.text = "$keyStr 번 ▼"
                dismiss()
            }
        }.show()
    }

    private fun refreshSpeedDialList() {
        listLayout.removeAllViews()
        val prefs = getSharedPreferences("SpeedDial", Context.MODE_PRIVATE)
        val sortedEntries = prefs.all.entries.sortedBy { it.key.toIntOrNull() ?: 9999 }

        if (sortedEntries.isEmpty()) {
            listLayout.addView(TextView(this).apply { text = "등록된 번호가 없다 이말이야."; setPadding(0, 20, 0, 20) })
            return
        }

        sortedEntries.forEach { entry ->
            val key = entry.key
            val rawNumber = entry.value.toString()
            val cleanNum = rawNumber.replace("-", "")

            val matchedContact = allContacts.find { it.phoneNumber.replace("-", "") == cleanNum }
            val name = matchedContact?.name

            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, 15, 0, 15)
                gravity = android.view.Gravity.CENTER_VERTICAL
            }

            // 단축번호 원형 아이콘
            val keyContainer = RelativeLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(50), dpToPx(50)).apply {
                    marginEnd = dpToPx(15)
                }
            }

            val keyBg = View(this).apply {
                layoutParams = RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.MATCH_PARENT, RelativeLayout.LayoutParams.MATCH_PARENT)
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.OVAL
                    setColor(android.graphics.Color.parseColor("#8E24AA"))
                }
            }
            keyContainer.addView(keyBg)

            val tvKey = TextView(this).apply {
                text = key
                setTextColor(android.graphics.Color.WHITE)
                textSize = 20f
                setTypeface(null, android.graphics.Typeface.BOLD)
                layoutParams = RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT).apply {
                    addRule(RelativeLayout.CENTER_IN_PARENT)
                }
            }
            keyContainer.addView(tvKey)
            row.addView(keyContainer)

            val textContainer = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }

            val tvName = TextView(this).apply {
                text = if (name.isNullOrEmpty()) rawNumber else name
                textSize = 18f
                setTextColor(android.graphics.Color.BLACK)
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            textContainer.addView(tvName)

            val tvNum = TextView(this).apply {
                text = rawNumber
                textSize = 14f
                setTextColor(android.graphics.Color.parseColor("#888888"))
                visibility = if (name.isNullOrEmpty()) View.GONE else View.VISIBLE
            }
            textContainer.addView(tvNum)
            row.addView(textContainer)

            row.addView(TextView(this).apply {
                text = "-"
                textSize = 40f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setTextColor(android.graphics.Color.parseColor("#FF3B30"))
                setPadding(20, 0, 20, 0)

                setOnClickListener {
                    prefs.edit().remove(key).apply()
                    Toast.makeText(this@SettingsSpeedDialActivity, "${key}번 성불 완료!", Toast.LENGTH_SHORT).show()
                    refreshSpeedDialList()
                }
            })
            listLayout.addView(row)
        }
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }
}