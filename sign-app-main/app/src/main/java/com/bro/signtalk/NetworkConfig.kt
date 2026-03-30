package com.bro.signtalk

object NetworkConfig {
    // 배포 서버 주소
    const val BASE_URL = "https://sign-call.p-e.kr"
    const val WS_URL   = "wss://sign-call.p-e.kr"

    // 로컬 테스트 시 아래로 교체 (PC IP로 바꿔서 사용)
    // const val BASE_URL = "http://192.168.x.x:3000"
    // const val WS_URL   = "ws://192.168.x.x:3000"
}
