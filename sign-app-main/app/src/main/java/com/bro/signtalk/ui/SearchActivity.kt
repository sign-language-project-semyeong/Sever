package com.bro.signtalk.ui

import android.content.Context
import android.os.Bundle
import android.provider.CallLog
import android.provider.ContactsContract
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.bro.signtalk.R
import com.bro.signtalk.data.model.CallType
import com.bro.signtalk.data.model.Contact
import com.bro.signtalk.data.model.RecentCall
import com.bro.signtalk.util.SearchHistoryManager

class SearchActivity : AppCompatActivity() {
    private var callLogOffset = 0 // [핵심] 몇 번째부터 가져올지 결정하는 서열 번호!
    private var currentSearchQuery = "" // 지금 무슨 검색 중인지 기억해라!
    private lateinit var searchAdapter: SearchAdapter
    private lateinit var searchInput: EditText
    private lateinit var rvResults: RecyclerView
    private lateinit var tvNoResult: TextView

    // [핵심] 여기에 주소록 데이터를 찰@지게 채워넣어야 한다 이말이야!
    private var allContacts = mutableListOf<Contact>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_search)

        searchInput = findViewById(R.id.et_search_bar)
        rvResults = findViewById(R.id.rv_search_results)
        tvNoResult = findViewById(R.id.tv_no_result)

        // 1. [필살기] 주소록 데이터부터 0.1초 만에 털어와라!
        loadContactsFromSystem()

        // 2. 어댑터 서열 정리
        searchAdapter = SearchAdapter(
            onHistoryClick = { history -> executeSearch(history) },
            onHistoryDelete = { history ->
                SearchHistoryManager.removeSearchQuery(this, history)
                showSearchHistory()
            }
        )
        rvResults.adapter = searchAdapter
        rvResults.layoutManager = LinearLayoutManager(this)

        // 3. [쫀득] 실시간 텍스트 리스너 기강 잡기
        searchInput.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                val query = s?.toString()?.trim() ?: ""
                if (query.isEmpty()) {
                    showSearchHistory()
                } else {
                    performSearch(query) // 글자 바뀔 때마다 0.1초 만에 검색!
                }
            }
            override fun beforeTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {}
            override fun onTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {}
        })

        // SearchActivity.kt onCreate 내부
        rvResults.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                // 아래로 스크롤 중이고 바닥(1)에 닿았냐?!?
                if (dy > 0 && !recyclerView.canScrollVertically(1)) {
                    loadMoreCallLogs()
                }
            }
        })

        // 초기 화면은 검색 기록!
        showSearchHistory()
    }

    // [쌈뽕] 시스템 주소록 털어오는 함수다 유남생?!?
    private fun loadContactsFromSystem() {
        allContacts.clear() // [필살기] 로딩 전에 싹 비워야 중복이 성불한다!
        val cursor = contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            null, null, null,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " ASC"
        )

        cursor?.use {
            val nameIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
            val numberIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
            val idIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.CONTACT_ID)

            while (it.moveToNext()) {
                val name = it.getString(nameIndex) ?: ""
                val number = it.getString(numberIndex) ?: ""
                val id = it.getString(idIndex) ?: ""

                // [쫀득] 번호가 같으면 한 놈만 남기고 컷해라!
                if (allContacts.none { it.phoneNumber == number }) {
                    allContacts.add(Contact(id, name, number))
                }
            }
        }
    }

    // ... 기존 executeSearch, showSearchHistory, hideKeyboard, getRecentCalls 함수 유지 ...

    // SearchActivity.kt 내부의 performSearch 함수를 이@렇게 갈아엎어라!
    private fun executeSearch(query: String) {
        searchInput.setText(query)
        searchInput.setSelection(query.length)
        SearchHistoryManager.addSearchQuery(this, query) // 기록 저장 똬악!
        performSearch(query)
        hideKeyboard()
    }

    // SearchActivity.kt 수정본

    private fun performSearch(query: String) {
        currentSearchQuery = query
        callLogOffset = 0

        // [쫀득] 사용자가 친 검색어에서 하이픈 싹 다 걷어내라 팍씨!
        val cleanQuery = query.replace("-", "")

        val filtered = allContacts.filter {
            // [핵심] 주소록 번호에서도 하이픈 빼고 알맹이끼리만 비교하는 마법이다 이말이야!
            val cleanPhoneNumber = it.phoneNumber.replace("-", "")

            it.name.contains(query, ignoreCase = true) || cleanPhoneNumber.contains(cleanQuery)
        }

        if (filtered.isEmpty()) {
            rvResults.visibility = View.GONE
            tvNoResult.visibility = View.VISIBLE
            return
        }

        val sortedResults = filtered.sortedWith(compareBy<Contact> {
            val index = it.name.indexOf(query, ignoreCase = true)
            if (index == -1) 999 else index
        }.thenBy { it.name })

        tvNoResult.visibility = View.GONE
        rvResults.visibility = View.VISIBLE

        val topMatch = sortedResults.first()
        val recentCalls = getRecentCallsObjects(topMatch.phoneNumber, 20)

        searchAdapter.updateData(sortedResults, recentCalls)
    }

    // [쌈뽕] 기존 최근기록창 로직을 그대로 가져온 쿼리문이다 이말이야!
    // SearchActivity.kt 내부의 getRecentCallsObjects 수정본

    // SearchActivity.kt 수정본


    private fun loadMoreCallLogs() {
        if (currentSearchQuery.isEmpty()) return

        callLogOffset += 20 // 0.1초 만에 오프셋 증가!
        val topMatchNumber = allContacts.find {
            it.name.contains(currentSearchQuery, ignoreCase = true)
        }?.phoneNumber ?: return

        val moreLogs = getRecentCallsObjects(topMatchNumber)
        if (moreLogs.isNotEmpty()) {
            searchAdapter.addMoreCalls(moreLogs) // 어댑터 끝에 찰지게 붙여라!
        }
    }

    private fun getRecentCallsObjects(phoneNumber: String, limit: Int = 20): List<RecentCall> {
        val logs = mutableListOf<RecentCall>()
        val cleanNumber = phoneNumber.replace("-", "")

        val cursor = contentResolver.query(
            CallLog.Calls.CONTENT_URI,
            null,
            "REPLACE(${CallLog.Calls.NUMBER}, '-', '') = ?",
            arrayOf(cleanNumber),
            "${CallLog.Calls.DATE} DESC LIMIT $limit OFFSET $callLogOffset" // [쌈뽕] 페이징 적용!
        )

        cursor?.use {
            val numberIdx = it.getColumnIndex(CallLog.Calls.NUMBER)
            val dateIdx = it.getColumnIndex(CallLog.Calls.DATE)
            val typeIdx = it.getColumnIndex(CallLog.Calls.TYPE)

            while (it.moveToNext()) {
                val rawTimestamp = it.getLong(dateIdx)
                val number = it.getString(numberIdx) ?: ""

                // [팩폭] 번호로 내 연락처에서 이름을 0.1초 만에 찾아라! 그래야 번호 대신 이름이 나온다 유남생?!?
                val matchedName = allContacts.find {
                    it.phoneNumber.replace("-", "") == number.replace("-", "")
                }?.name ?: number

                logs.add(RecentCall(
                    phoneNumber = number,
                    name = matchedName, // [쫀득] 이제 번호 대신 이름이 똬악!
                    callTime = DateUtils.formatDateTime(rawTimestamp),
                    dateGroup = DateUtils.getDateGroupName(rawTimestamp),
                    timestamp = rawTimestamp,
                    type = when(it.getInt(typeIdx)) {
                        CallLog.Calls.INCOMING_TYPE -> CallType.INCOMING
                        CallLog.Calls.OUTGOING_TYPE -> CallType.OUTGOING
                        else -> CallType.MISSED
                    }
                ))
            }
        }
        return logs
    }


    // [팩폭] 번호로 통화기록 캐내는 쌈뽕한 쿼리문이다!


    private fun showSearchHistory() {
        val history = SearchHistoryManager.getHistory(this)
        tvNoResult.visibility = View.GONE
        rvResults.visibility = View.VISIBLE
        // 어댑터를 기록 모드로 설정
        // searchAdapter.setHistoryMode(history)
    }

    // ... 기존 showSearchHistory, hideKeyboard 함수 유지 ...

    private fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(searchInput.windowToken, 0)
    }

    // [핵심] XML에서 애타게 찾던 그 함수다 팍씨! 이거 없어서 앱이 죽은 거다!
    fun onBackButtonClicked(view: View) {
        hideKeyboard() // [쫀득] 닫기 전에 키보드 싹바가지 없게 떠 있는 거 닫아주고!
        finish() // [쌈뽕] 뒤로 가기 눌렀으니까 화면 찰지게 닫아라 이말이야!
    }
}