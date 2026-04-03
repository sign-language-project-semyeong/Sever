package com.bro.signtalk.ui

import com.bro.signtalk.MainActivity
import android.transition.TransitionManager
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.bro.signtalk.R
import com.bro.signtalk.data.model.CallType
import com.bro.signtalk.data.model.Contact
import com.bro.signtalk.data.model.RecentCall

class SearchAdapter(
    private val onHistoryClick: (String) -> Unit,
    private val onHistoryDelete: (String) -> Unit
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private var items = mutableListOf<Any>()

    companion object {
        private const val TYPE_HEADER = 0
        private const val TYPE_CONTACT = 1
        private const val TYPE_CALL_LOG = 2
        private const val TYPE_HISTORY = 3
    }

    // [쌈뽕] 검색 결과 모드 (연락처 + 통화내역)
    fun updateData(contacts: List<Contact>, calls: List<RecentCall>) {
        items.clear()
        if (contacts.isNotEmpty()) {
            items.add("HEADER:연락처")
            items.addAll(contacts)
        }
        if (calls.isNotEmpty()) {
            items.add("HEADER:최근 통화 기록")
            items.addAll(calls)
        }
        notifyDataSetChanged()
    }

    // [쫀득] 검색 기록 모드 (기록 리스트)
    fun setHistoryMode(history: List<String>) {
        items.clear()
        if (history.isNotEmpty()) {
            items.add("HEADER:최근 검색어")
            items.addAll(history)
        }
        notifyDataSetChanged()
    }

    override fun getItemViewType(position: Int): Int = when (val item = items[position]) {
        is String -> if (item.startsWith("HEADER:")) TYPE_HEADER else TYPE_HISTORY
        is Contact -> TYPE_CONTACT
        is RecentCall -> TYPE_CALL_LOG
        else -> TYPE_HISTORY
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when (viewType) {
            TYPE_HEADER -> HeaderViewHolder(inflater.inflate(R.layout.item_search_header, parent, false))
            TYPE_CONTACT -> ContactViewHolder(inflater.inflate(R.layout.item_contact, parent, false))
            TYPE_CALL_LOG -> CallLogViewHolder(inflater.inflate(R.layout.item_recent_call, parent, false))
            else -> HistoryViewHolder(inflater.inflate(R.layout.item_search_history, parent, false))
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val item = items[position]
        when (holder) {
            is HeaderViewHolder -> holder.bind(item.toString().replace("HEADER:", ""))
            is ContactViewHolder -> holder.bind(item as Contact)
            is CallLogViewHolder -> holder.bind(item as RecentCall)
            is HistoryViewHolder -> holder.bind(item as String, onHistoryClick, onHistoryDelete)
        }
    }

    override fun getItemCount(): Int = items.size

    // 1. [헤더] 섹션 나누는 바
    class HeaderViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        fun bind(title: String) {
            itemView.findViewById<TextView>(R.id.tv_header_title).text = title
        }
    }

    // 2. [연락처] 검색된 연락처
    // SearchAdapter.kt 내부 ContactViewHolder 수정

    class ContactViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private var isExpanded = false

        fun bind(contact: Contact) {
            val tvName = itemView.findViewById<TextView>(R.id.tv_contact_name)
            val tvNumber = itemView.findViewById<TextView>(R.id.tv_contact_number)
            val layoutExpand = itemView.findViewById<View>(R.id.layout_expand)

            tvName.text = contact.name
            tvNumber.text = contact.phoneNumber

            // 1. [쫀득] 갤럭시 감성 찰진 슬라이드 애니메이션!
            itemView.setOnClickListener {
                isExpanded = !isExpanded
                // [필살기] 부모 뷰 멱살 잡고 변화를 주라고 시켜라! 유남생?!?
                TransitionManager.beginDelayedTransition(itemView.parent as ViewGroup)
                layoutExpand.visibility = if (isExpanded) View.VISIBLE else View.GONE
            }

            // 2. [쌈뽕] 버튼 연결 (새로 고친 비서실 호출!)
            itemView.findViewById<View>(R.id.btn_item_call).setOnClickListener {
                CallNavigation.makeVoiceCall(itemView.context, contact.phoneNumber)
            }
            itemView.findViewById<View>(R.id.btn_item_video).setOnClickListener {
                CallNavigation.makeVideoCall(itemView.context, contact.phoneNumber)
            }
            itemView.findViewById<View>(R.id.btn_item_message).setOnClickListener {
                CallNavigation.sendSms(itemView.context, contact.phoneNumber)
            }
        }
    }

    class CallLogViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        fun bind(call: RecentCall) {
            val tvName = itemView.findViewById<TextView>(R.id.tv_name)
            val tvDate = itemView.findViewById<TextView>(R.id.tv_contact_time)
            val ivType = itemView.findViewById<ImageView>(R.id.iv_call_type)

            // 4. [팩폭] 번호 대신 이름을 찍으라고 팍@씨! 유남생?!?
            // 이름 없으면 번호라도 보여주는 게 싹바가지 있는 UI다!
            tvName.text = if (call.name.isNotEmpty()) call.name else call.phoneNumber
            tvDate.text = DateUtils.getDateGroupName(call.timestamp)

            when(call.type) {
                CallType.INCOMING -> ivType.setImageResource(R.drawable.ic_call_incoming)
                CallType.OUTGOING -> ivType.setImageResource(R.drawable.ic_call_outgoing)
                CallType.MISSED -> ivType.setImageResource(R.drawable.ic_call_missed)
            }
        }
    }

    // 3. [통화기록] 기존 최근기록 레이아웃 그대로 사용!
    // SearchAdapter.kt 내부의 CallLogViewHolder 수정본

    // SearchAdapter.kt 내부에 추가!
    fun addMoreCalls(moreCalls: List<RecentCall>) {
        if (moreCalls.isEmpty()) return

        // [쫀득] 기존 리스트 끝에 새로운 통화 기록을 찰지게 붙여라!
        val startPosition = items.size
        items.addAll(moreCalls)

        // [필살기] 전체 갱신 말고 바뀐 부분만 0.1초 만에 알려줘라!
        notifyItemRangeInserted(startPosition, moreCalls.size)
    }


    // 4. [검색기록] X 버튼 있는 놈
    class HistoryViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        fun bind(text: String, onClick: (String) -> Unit, onDelete: (String) -> Unit) {
            itemView.findViewById<TextView>(R.id.tv_history_text).text = text
            itemView.setOnClickListener { onClick(text) }
            itemView.findViewById<ImageButton>(R.id.btn_delete_history).setOnClickListener { onDelete(text) }
        }
    }
}
