package com.bro.signtalk.ui.recent

import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.bro.signtalk.R

// RecentCallsAdapter.kt 하단부 수정
class RecentCallViewHolder(view: View) : RecyclerView.ViewHolder(view) {
    val tvName: TextView = view.findViewById(R.id.tv_name)
    val tvTime: TextView = view.findViewById(R.id.tv_contact_time)

    // [팩폭] 이 녀석 없으면 아이콘 못 바꾼다! XML ID랑 맞춰라!
    val ivCallType: android.widget.ImageView = view.findViewById(R.id.iv_call_type)

    val layoutExpand: View = view.findViewById(R.id.layout_expand)
    val btnCall: View = view.findViewById(R.id.btn_item_call)
    val btnMessage: View = view.findViewById(R.id.btn_item_message)
    val btnVideo: View = view.findViewById(R.id.btn_item_video)
}