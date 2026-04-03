package com.bro.signtalk.ui.custom

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class SideIndexView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val indexList = arrayOf("★", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "#")
    private val paint = Paint().apply {
        color = Color.GRAY
        textSize = 30f
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
    }

    private var indices: List<String> = emptyList()

    // [팩폭] 이게 없어서 에러 난 거다! 외부에서 존재하는 초성만 넘겨주는 입구!
    fun setIndices(newIndices: List<String>) {
        this.indices = newIndices
        // [필살기] 데이터 바뀌었으면 다시 그려야지 유남생?!?
        invalidate()
        requestLayout() // [핵심] 이거 있어야 높이 다시 계산해서 똬악 나타난다!
    }
    // 1. [필살기] 내 몸집(높이)을 스스로 결정하는 로직이다 이말이야!
    // 1. [필살기] 황@금 비율 높이 계산 드간다!
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val drawList = if (indices.isNotEmpty()) indices else indexList.toList()
        val density = context.resources.displayMetrics.density

        // [필살기] 기존 38dp에서 26dp로 슬림하게 다이어트!
        // 그래야 리스트 시야를 0.1초 만에 확보한다 이말이야!
        val itemSize = (26 * density).toInt()

        val desiredHeight = drawList.size * itemSize
        // [쫀득] 너비도 35dp에서 24dp로 줄여서 공간 낭비 성불시켜라!
        val desiredWidth = (24 * density).toInt()

        setMeasuredDimension(desiredWidth, desiredHeight)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val drawList = if (indices.isNotEmpty()) indices else indexList.toList()
        if (drawList.isEmpty()) return

        // [팩폭] 바가 슬림해졌으니 글자 크기도 35f에서 28f 정도로 0.1초 만에 다이어트!
        paint.color = Color.parseColor("#880E4F") // 딥 마젠타 포인트!
        paint.textSize = 28f
        paint.isFakeBoldText = true // [쫀득] 얇아진 대신 두껍게 해서 가독성 챙겨라!

        val itemHeight = height.toFloat() / drawList.size

        for (i in drawList.indices) {
            val xPos = width / 2f
            // [필살기] 텍스트가 칸 정중앙에 똬악 박히게 하는 마법의 공식!
            val yPos = (itemHeight * i) + (itemHeight / 2f) - ((paint.descent() + paint.ascent()) / 2f)
            canvas.drawText(drawList[i], xPos, yPos, paint)
        }
    }



    override fun onTouchEvent(event: MotionEvent): Boolean {
        // [핵심] 그릴 때 쓴 리스트랑 터치 계산할 리스트가 같아야 한다!
        val drawList = if (indices.isNotEmpty()) indices else indexList.toList()
        if (drawList.isEmpty()) return false

        val itemHeight = height.toFloat() / drawList.size
        val index = (event.y / itemHeight).toInt().coerceIn(0, drawList.size - 1)

        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                // [찰진] 리스너 호출!
                onIndexTouchListener?.invoke(drawList[index], true)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                onIndexTouchListener?.invoke(drawList[index], false)
            }
        }
        return true
    }

    // 리스너 정의도 살짝 바꿔라 브로! (글자, 표시여부)
    var onIndexTouchListener: ((String, Boolean) -> Unit)? = null
}

