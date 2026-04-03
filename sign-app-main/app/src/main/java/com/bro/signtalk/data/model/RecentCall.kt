package com.bro.signtalk.data.model

import java.io.Serializable

// [팩폭] 파라미터 순서랑 타입이 이@렇게 되어 있어야 한다 이말이야!
data class RecentCall(
    val phoneNumber: String,
    val name: String,
    val callTime: String,
    val type: CallType,
    val dateGroup: String,
    val timestamp: Long, // [필살기] 진짜 DB 지울 때 쓰는 핵심 키다!
    var isBlocked: Boolean = false
)