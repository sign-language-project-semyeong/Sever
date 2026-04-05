package com.bro.signtalk

object NetworkConfig {
    // Local test
    private const val SERVER_IP = "192.168.123.100"
    const val BASE_URL = "http://$SERVER_IP:3001"
    const val WS_URL   = "ws://$SERVER_IP:3001"
}
