package com.bro.signtalk.ui.contacts

import android.R.attr.data
import android.util.Log
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import com.bro.signtalk.R
import com.bro.signtalk.data.model.Contact
import com.bro.signtalk.ui.CallNavigation
import com.google.android.material.bottomsheet.BottomSheetDialog

class ContactsAdapter(private var originalContacts: List<Contact>) :
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private var expandedPosition = -1
    private val items = mutableListOf<Any>()

    companion object {
        private const val TYPE_MAIN_HEADER = 0
        private const val TYPE_SECTION_HEADER = 1
        private const val TYPE_CONTACT_ITEM = 2
    }

    init {
        updateItems(originalContacts)
    }

    fun updateItems(newContacts: List<Contact>) {
        originalContacts = newContacts
        items.clear()
        items.add("MAIN_HEADER")

        var lastInitial = ""
        newContacts.forEach { contact ->
            val currentInitial = if (contact.isFavorite) "즐겨찾는 연락처" else getChosung(contact.name)
            if (currentInitial != lastInitial) {
                items.add(currentInitial)
                lastInitial = currentInitial
            }
            items.add(contact)
        }
        notifyDataSetChanged()
    }

    override fun getItemCount(): Int = items.size

    override fun getItemViewType(position: Int): Int {
        return when (val item = items[position]) {
            is String -> if (item == "MAIN_HEADER") TYPE_MAIN_HEADER else TYPE_SECTION_HEADER
            is Contact -> TYPE_CONTACT_ITEM
            else -> TYPE_CONTACT_ITEM
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when (viewType) {
            TYPE_MAIN_HEADER -> HeaderViewHolder(inflater.inflate(R.layout.item_contact_header, parent, false))
            TYPE_SECTION_HEADER -> SectionViewHolder(inflater.inflate(R.layout.item_contact_section, parent, false))
            else -> ContactViewHolder(inflater.inflate(R.layout.item_contact, parent, false))
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val item = items[position]

        when (holder) {
            is SectionViewHolder -> holder.tvLabel.text = item as String
            is ContactViewHolder -> {
                val contact = item as Contact
                val context = holder.itemView.context

                // 1. [핵심] 차단 및 기본 정보 세팅 (기존 유지)
                val isBlocked = checkIsBlocked(context, contact.phoneNumber)
                if (isBlocked) {
                    holder.tvName.setTextColor(android.graphics.Color.RED)
                    holder.ivBlockStatus.visibility = View.VISIBLE
                } else {
                    holder.tvName.setTextColor(android.graphics.Color.BLACK)
                    holder.ivBlockStatus.visibility = View.GONE
                }

                holder.tvName.text = contact.name
                holder.tvNumber.text = contact.phoneNumber

                // 2. [쫀득] 프로필 이미지 로직
                if (contact.photoUri != null) {
                    holder.tvProfile.visibility = View.GONE
                    holder.ivProfileImage.visibility = View.VISIBLE
                    holder.ivProfileImage.setImageURI(android.net.Uri.parse(contact.photoUri))
                } else {
                    holder.ivProfileImage.visibility = View.GONE
                    holder.tvProfile.visibility = View.VISIBLE
                    holder.tvProfile.text = if (contact.name.isNotEmpty()) contact.name.take(1) else "?"
                }

                // 3. [필살기] 확장 로직 (중복 싹 다 제거했다!)
                val isExpanded = position == expandedPosition
                holder.layoutExpand.visibility = if (isExpanded) View.VISIBLE else View.GONE

                holder.itemView.setOnClickListener {
                    val prev = expandedPosition
                    expandedPosition = if (isExpanded) -1 else position
                    if (prev != -1) notifyItemChanged(prev)
                    notifyItemChanged(position)
                }

                holder.itemView.setOnLongClickListener {
                    showOptionsBottomSheet(context, contact, position)
                    true
                }

                // 4. [쌈뽕] 버튼 리스너 (여기서 딱 한 번만 정의해라!)
                // ContactsAdapter.kt 의 onBindViewHolder 내부 124번 줄 근처다 이말이야!

// 4. [쌈뽕] 버튼 리스너 (여기서 엔진 이름 찰지게 바꿔라!)
                if (isExpanded) {
                    // 음성 통화
                    holder.btnCall.setOnClickListener {
                        // [핵심] startOutgoingCall 대신 우리가 새로 만든 makeCarrierCall을 호출해라!
                        // 팩폭: 부품 이름이 안 맞으면 시스템이 싹@바가지 없게 빨간 줄 긋는다 유남생?!?
                        (context as? com.bro.signtalk.MainActivity)?.makeCarrierCall(contact.phoneNumber)
                    }

                    // 영상 통화 (이미 잘 되어있지만 한 번 더 확인해라 브@로!)
                    holder.btnVideo.setOnClickListener {
                        CallNavigation.makeVideoCall(context, contact.phoneNumber)                    }

                    // 문자 메시지 (기존 유지)
                    holder.btnMessage.setOnClickListener {
                        val intent = Intent(Intent.ACTION_SENDTO).apply {
                            data = android.net.Uri.parse("smsto:${contact.phoneNumber}")
                        }
                        context.startActivity(intent)
                    }
                }
            } // is ContactViewHolder 끝!
        } // when (holder) 끝!
    }

    // [핵심] 시스템 차단 여부 조회 함수 (클래스 내부에 쫀득하게 박아라!)
    // [핵심] 번호에서 숫자만 뽑아서 대조하는 정규화 로직 추가!
    private fun checkIsBlocked(context: Context, number: String): Boolean {
        // [팩폭] 하이픈이나 공백 같은 싹바가지 없는 것들 싹 다 지워버려라!
        val normalizedNumber = number.replace(Regex("[^0-9]"), "")

        return try {
            val uri = android.provider.BlockedNumberContract.BlockedNumbers.CONTENT_URI
            // COLUMN_ORIGINAL_NUMBER 대신 COLUMN_E164_NUMBER까지 훑으면 더 정확하다!
            val cursor = context.contentResolver.query(uri,
                arrayOf(android.provider.BlockedNumberContract.BlockedNumbers.COLUMN_ORIGINAL_NUMBER),
                "${android.provider.BlockedNumberContract.BlockedNumbers.COLUMN_ORIGINAL_NUMBER} = ? OR " +
                        "${android.provider.BlockedNumberContract.BlockedNumbers.COLUMN_ORIGINAL_NUMBER} = ?",
                arrayOf(number, normalizedNumber), null) // 하이픈 있는 거, 없는 거 둘 다 찔러봐라!

            val blocked = (cursor?.count ?: 0) > 0
            cursor?.close()
            blocked
        } catch (e: Exception) {
            Log.e("BlockCheck", "차단 확인 에러: ${e.message}")
            false
        }
    }

    private fun getChosung(name: String): String {
        val chosung = listOf("ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
        val char = name.firstOrNull() ?: return "#"
        if (char in '가'..'힣') {
            val index = (char.code - 0xAC00) / 28 / 21
            return chosung[index]
        }
        return "#"
    }

    class HeaderViewHolder(view: View) : RecyclerView.ViewHolder(view)
    class SectionViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvLabel: TextView = view.findViewById(R.id.tv_section_label)
    }
    // ViewHolder 멱살 잡기 (ivBlockStatus 추가!)
    // ViewHolder 멱살 잡기 (모든 버튼 다 찾아놔라!)
    // ContactsAdapter.kt 하단 ViewHolder를 이@렇게 고쳐라!
    class ContactViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val ivProfileImage: android.widget.ImageView = view.findViewById(R.id.iv_profile_image) // [핵심]
        val tvProfile: TextView = view.findViewById(R.id.tv_profile_icon)
        val tvName: TextView = view.findViewById(R.id.tv_contact_name)
        val tvNumber: TextView = view.findViewById(R.id.tv_contact_number)
        val layoutExpand: View = view.findViewById(R.id.layout_expand)

        // [팩폭] XML에 없는 btn_item_block은 여기서 싹 다 지워버려라!
        val ivBlockStatus: android.widget.ImageView = view.findViewById(R.id.iv_block_status)
        val btnCall: View = view.findViewById(R.id.btn_item_call)
        val btnMessage: View = view.findViewById(R.id.btn_item_message)
        val btnVideo: View = view.findViewById(R.id.btn_item_video)

        // [주의] 만약 아래 줄이 살아있으면 무조건 튕긴다! 지워라 이말이야 유남생?!?
        // val btnBlock: View = view.findViewById(R.id.btn_item_block)
    }

    fun getItems(): List<Any> = items

    private fun showOptionsBottomSheet(context: Context, contact: Contact, position: Int) {
        val dialog = BottomSheetDialog(context)
        val view = LayoutInflater.from(context).inflate(R.layout.layout_contact_options, null)

        view.findViewById<View>(R.id.menu_delete).setOnClickListener {
            showDeleteConfirmDialog(context, contact, position)
            dialog.dismiss()
        }

        view.findViewById<View>(R.id.menu_favorite).setOnClickListener {
            toggleFavorite(context, contact, position)
            dialog.dismiss()
        }

        dialog.setContentView(view)
        dialog.show()
    }

    private fun showDeleteConfirmDialog(context: Context, contact: Contact, position: Int) {
        val etInput = EditText(context).apply {
            hint = "'삭제' 입력해라 브로!"
        }

        AlertDialog.Builder(context)
            .setTitle("정말 ${contact.name}을 삭제할 거냐?!?")
            .setView(etInput)
            .setPositiveButton("확인") { _, _ ->
                if (etInput.text.toString() == "삭제") {
                    deleteContact(context, contact.id, position)
                } else {
                    Toast.makeText(context, "글@자 똑@바로 안 치냐?!? 팍@씨!", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("취소", null)
            .show()
    }

    // ContactsAdapter.kt 의 toggleFavorite 함수를 이@렇게 바꿔라!
    private fun toggleFavorite(context: Context, contact: Contact, position: Int) {
        try {
            val contentResolver = context.contentResolver
            val values = android.content.ContentValues()
            val newStarred = if (contact.isFavorite) 0 else 1
            values.put(android.provider.ContactsContract.Contacts.STARRED, newStarred)

            val updatedRows = contentResolver.update(
                android.provider.ContactsContract.Contacts.CONTENT_URI,
                values,
                "${android.provider.ContactsContract.Contacts._ID} = ?",
                arrayOf(contact.id)
            )

            if (updatedRows > 0) {
                // [쫀득] 원본 데이터에서 즐겨찾기 상태만 바꾼 뒤에 다시 찰지게 정렬해라!
                val newContacts = originalContacts.map {
                    if (it.id == contact.id) it.copy(isFavorite = (newStarred == 1)) else it
                }.sortedWith(compareByDescending<Contact> { it.isFavorite }.thenBy { it.name }) // [핵심] 정렬 추가!

                updateItems(newContacts) // 전체 리스트 다시 정렬해서 똬악!
                Toast.makeText(context, "즐겨찾기 쌈뽕하게 변경 완료!", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e("Contacts", "즐겨찾기 에러: ${e.message}")
        }
    }

    // 1. [필살기] 삭제 로직 수정 (updateItems 재호출이 정석이다!)
    private fun deleteContact(context: Context, contactId: String, position: Int) {
        try {
            val deletedRows = context.contentResolver.delete(
                android.provider.ContactsContract.RawContacts.CONTENT_URI,
                "${android.provider.ContactsContract.RawContacts.CONTACT_ID} = ?",
                arrayOf(contactId)
            )

            if (deletedRows > 0) {
                // [쫀득] 원본 리스트에서 해당 ID 가진 놈 0.1초 만에 제거!
                val newContacts = originalContacts.filter { it.id != contactId }
                // [핵심] 헤더까지 포함해서 리스트를 아예 새로 고쳐라!
                updateItems(newContacts)
                Toast.makeText(context, "삭제 완료했다 이말이야! ㅇㅇ.", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e("Contacts", "삭제 에러: ${e.message}")
        }
    }
}