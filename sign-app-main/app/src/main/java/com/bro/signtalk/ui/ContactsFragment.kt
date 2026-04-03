package com.bro.signtalk.ui

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.ContactsContract
import android.telecom.TelecomManager
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.bro.signtalk.R
import com.bro.signtalk.data.model.Contact
import com.bro.signtalk.ui.contacts.ContactsAdapter
import com.bro.signtalk.ui.custom.SideIndexView

class ContactsFragment : Fragment() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var sideIndexView: SideIndexView
    private var tvOverlay: TextView? = null

    // [쫀득] 5초 타이머용 핸들러와 러너블
    private val hideHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val hideRunnable = Runnable {
        if (::sideIndexView.isInitialized) {
            sideIndexView.animate()
                .alpha(0f)
                .setDuration(300
                ).withEndAction {
                    sideIndexView.visibility = View.GONE
            }.start()
        }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.fragment_contacts, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        try {
            recyclerView = view.findViewById(R.id.contacts_recycler)
            sideIndexView = view.findViewById(R.id.side_index_view)
            tvOverlay = view.findViewById(R.id.tv_index_overlay)

            // 1. [필살기] 레이아웃 매니저랑 빈 어댑터를 미리 똬악 박아라!
            recyclerView.layoutManager = LinearLayoutManager(requireContext())
            recyclerView.adapter = ContactsAdapter(emptyList()) // 빈 리스트로 시작!

            sideIndexView.visibility = View.GONE
            sideIndexView.alpha = 0f

            // 2. [핵심] 리스너를 미리 달아놔야 어댑터가 바뀌어도 찰지게 감시한다!
            setupScrollListener()
            setupSideIndexListener()

            checkContactPermissions()

        } catch (e: Exception) {
            Log.e("Contacts", "초기화 중 튕김 발생: ${e.message}")
        }
    }

    private fun setupScrollListener() {
        recyclerView.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                super.onScrolled(recyclerView, dx, dy)

                if (java.lang.Math.abs(dy) > 1) {
                    // [필살기] 이미 나타나 있으면 애니메이션 또 하지 마! (색깔 춤추는 거 방지)
                    if (sideIndexView.visibility != View.VISIBLE || sideIndexView.alpha < 1f) {
                        sideIndexView.animate().cancel()
                        sideIndexView.visibility = View.VISIBLE

                        // [쫀득] 0.5f가 아니라 1f로 가야 XML의 #1A(10%)가 그대로 나온다 유남생?!?
                        sideIndexView.animate()
                            .alpha(1f)
                            .setDuration(150)
                            .setInterpolator(android.view.animation.DecelerateInterpolator())
                            .start()
                    }

                    hideHandler.removeCallbacks(hideRunnable)
                    hideHandler.postDelayed(hideRunnable, 3000)
                }
            }
        })
    }

    private fun setupSideIndexListener() {
        sideIndexView.onIndexTouchListener = { letter, isVisible ->
            hideHandler.removeCallbacks(hideRunnable)

            if (isVisible) {
                // [팩폭] 터치 중에도 1f를 유지해야 색깔이 안 변한다 이말이야!
                sideIndexView.animate().cancel()
                sideIndexView.alpha = 1f
                sideIndexView.visibility = View.VISIBLE

                tvOverlay?.text = letter
                tvOverlay?.visibility = View.VISIBLE

                val adapter = recyclerView.adapter as? ContactsAdapter
                val items = adapter?.getItems() ?: emptyList()

                val position = items.indexOfFirst { item ->
                    if (letter == "★") item is String && item == "즐겨찾는 연락처"
                    else item is String && item == letter
                }

                if (position != -1) {
                    val headerOffset = (30 * resources.displayMetrics.density).toInt()
                    (recyclerView.layoutManager as LinearLayoutManager)
                        .scrollToPositionWithOffset(position, headerOffset)
                }
            } else {
                tvOverlay?.visibility = View.GONE
                // [쫀득] 손 떼도 이미 1f니까 더 진해질 리가 없다 이말이야!
                hideHandler.postDelayed(hideRunnable, 2000)
            }
        }
    }
    // --- 나머지 로직 (displayContacts, fetchContacts 등 기존 유지) ---

    private fun displayContacts() {
        val contacts = fetchContacts()

        // 1. [핵심] 여기서 변수를 똬악 선언해라! (함수 내부)
        val existingIndices = mutableListOf<String>()

        // 즐겨찾기 있으면 별표 추가!
        if (contacts.any { it.isFavorite }) existingIndices.add("★")

        // 초성 추출해서 정렬 똬악!
        val chosungList = contacts.filter { !it.isFavorite }
            .map { getChosung(it.name) }
            .distinct()
            .sorted()

        existingIndices.addAll(chosungList)

        // 2. [필살기] 바로 여기서 사이드바에 값을 넣어라!
        // existingIndices가 살아있을 때 0.1초 만에 쌔려야 한다 이말이야!
        sideIndexView.setIndices(existingIndices)

        // 3. 어댑터 갱신 로직 (기존 유지)
        (recyclerView.adapter as? ContactsAdapter)?.let { adapter ->
            adapter.updateItems(contacts)
            Log.d("Contacts", "데이터랑 인덱스 쫀득하게 갱신 완료! ㅇㅇ.")
        }
    }

    private fun getChosung(name: String): String {
        val chosungList = listOf("ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
        val char = name.firstOrNull() ?: return "#"
        if (char in '가'..'힣') {
            val index = (char.code - 0xAC00) / 28 / 21
            return chosungList[index]
        }
        return "#"
    }

    private val requiredPermissions = arrayOf(
        Manifest.permission.READ_CONTACTS,
        Manifest.permission.WRITE_CONTACTS,
        Manifest.permission.CALL_PHONE,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.CAMERA
    )

    private fun checkContactPermissions() {
        val neededPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(requireContext(), it) != PackageManager.PERMISSION_GRANTED
        }
        if (neededPermissions.isNotEmpty()) {
            requestPermissions(neededPermissions.toTypedArray(), 100)
        } else {
            checkDefaultDialer()
            displayContacts()
        }
    }

    private fun checkDefaultDialer() {
        val telecomManager = requireContext().getSystemService(Context.TELECOM_SERVICE) as TelecomManager
        if (telecomManager.defaultDialerPackage != requireContext().packageName) {
            val intent = Intent(TelecomManager.ACTION_CHANGE_DEFAULT_DIALER).apply {
                putExtra(TelecomManager.EXTRA_CHANGE_DEFAULT_DIALER_PACKAGE_NAME, requireContext().packageName)
            }
            startActivityForResult(intent, 400)
        }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_CONTACTS)
            == PackageManager.PERMISSION_GRANTED) {
            displayContacts()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (requestCode == 100 && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            checkDefaultDialer()
            displayContacts()
        }
    }

    // [핵심] ContactsFragment.kt 내부의 fetchContacts() 함수를 이걸로 통째로 갈아 끼워라!
    private fun fetchContacts(): List<Contact> {
        val contactList = mutableListOf<Contact>()

        // [쫀득] Projection에 PHOTO_URI를 똬악 추가해서 프사까지 캐오라고 명령해라!
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.CONTACT_ID,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.STARRED,
            ContactsContract.CommonDataKinds.Phone.PHOTO_URI // [필살기] 프사 경로!
        )

        val cursor = requireContext().contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI, projection, null, null, null
        )

        cursor?.use {
            val nameIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
            val numberIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
            val starredIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.STARRED)
            val idIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.CONTACT_ID)
            val photoUriIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.PHOTO_URI) // 인덱스 찾기!

            while (it.moveToNext()) {
                val id = it.getString(idIndex) ?: ""
                val name = it.getString(nameIndex) ?: "이름 없음"
                val number = it.getString(numberIndex) ?: ""
                val isFavorite = it.getInt(starredIndex) > 0
                val photoUri = it.getString(photoUriIndex) // [쌈뽕] 경로 캐오기 완료!

                // Contact 모델에 photoUri까지 찰지게 담아서 리스트에 던져라!
                contactList.add(Contact(id, name, number, photoUri = photoUri, isFavorite = isFavorite))
            }
        }
        return contactList.sortedWith(compareByDescending<Contact> { it.isFavorite }.thenBy { it.name })
    }
}