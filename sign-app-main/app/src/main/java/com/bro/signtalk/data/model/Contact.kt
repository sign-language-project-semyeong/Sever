package com.bro.signtalk.data.model

// [팩폭] 여기에 isFavorite이 있어야 Fragment에서 불러다 쓸 수 있다 이말이야!
data class Contact(
    val id: String,
    val name: String,
    val phoneNumber: String,
    val photoUri: String? = null,
    val isFavorite: Boolean = false // [핵심] 이거 추가 안 하면 에러 파티다 유남생?!?
)