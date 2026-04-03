package com.bro.signtalk.ui

import java.util.*
import java.text.SimpleDateFormat

object DateUtils {
    // [필살기] 타임스탬프를 "오늘", "어제", "2026년 3월 30일"로 0.1초 만에 변환!
    fun getDateGroupName(timestamp: Long): String {
        val targetCal = Calendar.getInstance().apply { timeInMillis = timestamp }
        val nowCal = Calendar.getInstance() // [쫀득] 지금 이 순간의 시각!

        // 1. [오늘 검문] 연도랑 올해의 몇 번째 날인지가 똑같냐?!?
        val isToday = targetCal.get(Calendar.YEAR) == nowCal.get(Calendar.YEAR) &&
                targetCal.get(Calendar.DAY_OF_YEAR) == nowCal.get(Calendar.DAY_OF_YEAR)
        if (isToday) return "오늘"

        // 2. [어제 검문] 오늘에서 딱 하루 뺀 거랑 똑같냐?!?
        val yesterdayCal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -1) }
        val isYesterday = targetCal.get(Calendar.YEAR) == yesterdayCal.get(Calendar.YEAR) &&
                targetCal.get(Calendar.DAY_OF_YEAR) == yesterdayCal.get(Calendar.DAY_OF_YEAR)
        if (isYesterday) return "어제"

        // 3. [그 외] 그냥 날짜 찰지게 박아라!
        return SimpleDateFormat("yyyy년 M월 d일", Locale.KOREAN).format(targetCal.time)
    }
    fun formatDateTime(timestamp: Long): String {
        val date = Date(timestamp)
        // 오전/오후 구분을 위해 Locale.KOREAN 필수다 유남생?!?
        val sdf = SimpleDateFormat("a h:mm", Locale.KOREAN)
        return sdf.format(date)
    }
}