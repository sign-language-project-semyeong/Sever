package com.bro.signtalk.util

import android.content.Context
import org.json.JSONArray

object SearchHistoryManager {
    private const val PREFS_NAME = "search_prefs"
    private const val KEY_HISTORY = "history_list"
    private const val MAX_HISTORY_SIZE = 10



    fun getHistory(context: Context): MutableList<String> {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val jsonString = prefs.getString(KEY_HISTORY, "[]")
        val jsonArray = JSONArray(jsonString)
        val history = mutableListOf<String>()
        for (i in 0 until jsonArray.length()) {
            history.add(jsonArray.getString(i))
        }
        return history
    }

    fun addSearchQuery(context: Context, query: String) {
        if (query.isBlank()) return
        val history = getHistory(context)

        // 중복 제거: 기존에 같은 검색어가 있으면 지우고 맨 앞으로 가져옴
        history.remove(query)
        history.add(0, query)

        // 10개 초과 시 가장 오래된 것(마지막 항목) 삭제
        if (history.size > MAX_HISTORY_SIZE) {
            history.removeAt(history.size - 1)
        }

        saveHistory(context, history)
    }

    fun removeSearchQuery(context: Context, query: String) {
        val history = getHistory(context)
        history.remove(query)
        saveHistory(context, history)
    }

    private fun saveHistory(context: Context, history: List<String>) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val jsonArray = JSONArray()
        history.forEach { jsonArray.put(it) }
        prefs.edit().putString(KEY_HISTORY, jsonArray.toString()).apply()
    }
}