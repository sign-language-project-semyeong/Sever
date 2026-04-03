package com.bro.signtalk.ui.recent

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.CallLog
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.bro.signtalk.R
import com.bro.signtalk.data.model.CallType
import com.bro.signtalk.data.model.RecentCall
import com.bro.signtalk.ui.DateUtils

class RecentCallsFragment : Fragment() {
    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: RecentCallsAdapter
    private lateinit var deleteModeFrame: LinearLayout
    private lateinit var btnDeleteSelected: Button

    // [필살기] 뒤로가기 버튼을 가로채는 쌈뽕한 콜백이다 이말이야!
    private val backPressedCallback = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() {
            if (::adapter.isInitialized && adapter.isSelectMode) {
                adapter.exitSelectMode() // [쫀득] 냅다 끄지 말고 선택 모드만 예쁘게 종료시켜라!
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_recent_calls, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        recyclerView = view.findViewById(R.id.recent_calls_recycler)
        deleteModeFrame = view.findViewById(R.id.delete_mode_frame)
        btnDeleteSelected = view.findViewById(R.id.btn_delete_selected)

        // [핵심] 액티비티의 뒤로가기 시스템에 콜백을 똬악 달아놔라!
        requireActivity().onBackPressedDispatcher.addCallback(viewLifecycleOwner, backPressedCallback)

        deleteModeFrame.post {
            val bottomNav = requireActivity().findViewById<View>(R.id.bottom_navigation)
            if (bottomNav != null) {
                val params = deleteModeFrame.layoutParams as android.widget.FrameLayout.LayoutParams
                deleteModeFrame.layoutParams = params
            }
        }

        recyclerView.layoutManager = LinearLayoutManager(requireContext())
        checkPermissionsAndLoad()

        btnDeleteSelected.setOnClickListener {
            if (::adapter.isInitialized) {
                val selectedItems = adapter.selectedItems.toList()
                deleteSelectedCalls(selectedItems)
            }
        }
    }

    private fun setupAdapterListener() {
        adapter.selectListener = object : RecentCallsAdapter.OnSelectModeListener {
            override fun onStartSelectMode() {
                // 1. 뒤로가기 콜백 활성화! 이제부터 뒤로가기는 내가 통@제한다!
                backPressedCallback.isEnabled = true

                // 2. GONE 상태에서 벗어나게 해주고!
                deleteModeFrame.visibility = View.VISIBLE

                // 3. [팩폭] 뷰가 그려진 직후에(post) 애니메이션을 먹여야 첫 빵에도 안 씹히고 올라온다!
                deleteModeFrame.post {
                    deleteModeFrame.translationY = deleteModeFrame.height.toFloat()
                    deleteModeFrame.animate().translationY(0f).setDuration(300).start()
                }
            }

            override fun onSelectionChanged(count: Int) {
                btnDeleteSelected.text = "${count}개 기록 삭제"
            }

            override fun onEndSelectMode() {
                // 1. 볼일 끝났으면 뒤로가기 통제권 다시 돌@려주고!
                backPressedCallback.isEnabled = false

                // 2. 스르륵 바닥으로 꺼지게 찰지게 밀어 넣어라!
                deleteModeFrame.animate()
                    .translationY(deleteModeFrame.height.toFloat())
                    .setDuration(300)
                    .withEndAction {
                        deleteModeFrame.visibility = View.GONE
                    }.start()
            }
        }
    }

    private fun deleteSelectedCalls(selectedCalls: List<RecentCall>) {
        if (selectedCalls.isEmpty()) return

        var deletedCount = 0
        for (call in selectedCalls) {
            val selection = "${CallLog.Calls.NUMBER} = ? AND ${CallLog.Calls.DATE} = ?"
            val selectionArgs = arrayOf(call.phoneNumber, call.timestamp.toString())

            deletedCount += requireContext().contentResolver.delete(
                CallLog.Calls.CONTENT_URI,
                selection,
                selectionArgs
            )
        }

        if (deletedCount > 0) {
            Toast.makeText(requireContext(), "${deletedCount}개 기록 찰@지게 삭제 완료!", Toast.LENGTH_SHORT).show()
            loadRecentCalls()
            adapter.exitSelectMode()
        } else {
            Toast.makeText(requireContext(), "삭제 실@패했다 이말이야!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun checkPermissionsAndLoad() {
        // [팩폭] 지우려면 WRITE_CALL_LOG 권한도 필요하다! 같이 검문해라!
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_CALL_LOG) == PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.WRITE_CALL_LOG) == PackageManager.PERMISSION_GRANTED) {

            val realData = fetchCallLogs()
            adapter = RecentCallsAdapter(realData)
            recyclerView.adapter = adapter

            // [핵심] 길게 눌렀을 때 버튼 프레임 띄우는 리스너 연결!
            setupAdapterListener()
        } else {
            requestPermissions(arrayOf(Manifest.permission.READ_CALL_LOG, Manifest.permission.WRITE_CALL_LOG), 200)
        }
    }



    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 200 && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            val realData = fetchCallLogs()
            adapter = RecentCallsAdapter(realData)
            recyclerView.adapter = adapter
            setupAdapterListener()
        }
    }

    override fun onResume() {
        super.onResume()
        if (::adapter.isInitialized) {
            // [쫀득] 삭제 하려고 선택 중인데 화면 갱신돼서 초기화되는 거 막아라!
            if (!adapter.isSelectMode) {
                loadRecentCalls()
                Log.d("RecentCalls", "화면 복귀! 캘린더 날짜 갱신 완료! ㅇㅇ.")
            }
        }
    }

    private fun loadRecentCalls() {
        val calls = fetchCallLogs()
        adapter.updateItems(calls)
    }

    // fetchCallLogs 함수는 브@로가 짠 거 그대로 유지하면 된다 이말이야!
    private fun fetchCallLogs(): List<RecentCall> {
        val callList = mutableListOf<RecentCall>()
        val cursor = try {
            requireContext().contentResolver.query(
                android.provider.CallLog.Calls.CONTENT_URI,
                null, null, null,
                "${android.provider.CallLog.Calls.DATE} DESC"
            )
        } catch (e: SecurityException) { return emptyList() }

        cursor?.use {
            val nameIndex = it.getColumnIndex(android.provider.CallLog.Calls.CACHED_NAME)
            val numberIndex = it.getColumnIndex(android.provider.CallLog.Calls.NUMBER)
            val typeIndex = it.getColumnIndex(android.provider.CallLog.Calls.TYPE)
            val dateIndex = it.getColumnIndex(android.provider.CallLog.Calls.DATE)

            while (it.moveToNext()) {
                val name = if (nameIndex != -1) it.getString(nameIndex) ?: "" else ""
                val number = if (numberIndex != -1) it.getString(numberIndex) ?: "번호 없음" else "번호 없음"
                val typeRaw = if (typeIndex != -1) it.getInt(typeIndex) else 1
                val timestamp = if (dateIndex != -1) it.getLong(dateIndex) else System.currentTimeMillis()

                val callType = when (typeRaw) {
                    android.provider.CallLog.Calls.INCOMING_TYPE -> CallType.INCOMING
                    android.provider.CallLog.Calls.OUTGOING_TYPE -> CallType.OUTGOING
                    android.provider.CallLog.Calls.MISSED_TYPE -> CallType.MISSED
                    else -> CallType.INCOMING
                }

                val dateGroup = DateUtils.getDateGroupName(timestamp)

                callList.add(RecentCall(
                    phoneNumber = number,
                    name = name,
                    callTime = java.text.SimpleDateFormat("HH:mm", java.util.Locale.KOREAN).format(timestamp),
                    type = callType,
                    dateGroup = dateGroup,
                    timestamp = timestamp
                ))
            }
        }
        return callList
    }
}