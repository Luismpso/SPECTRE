package io.github.luismpso.spectre.core

import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Sample rate of everything in the core: 16 kHz mono. */
const val SR = 16000
private const val STEP = 0.01                       // voice activity detection: 25 ms frames every 10 ms

/** A finished turn: 16 kHz audio, seconds of detected speech and the short pauses inside it (sample indices). */
class Turn(val start: Double, val end: Double, val audio: FloatArray, val speechS: Double, val cuts: List<Int> = emptyList())

/**
 * Streaming energy-based voice activity detection that cuts the audio into turns at pauses — a port of
 * TurnSegmenter in spectre.conversation. A turn ends after [gapS] of silence; a long one at the first short pause
 * after [softMaxS], and never later than [maxTurnS]. Shorter pauses inside a turn are kept as possible cuts.
 */
class TurnSegmenter(
    gapS: Double = 0.6, softMaxS: Double = 6.0, softGapS: Double = 0.25, maxTurnS: Double = 15.0,
    cutGapS: Double = 0.2, private val minSpeechS: Double = 0.3, private val marginDb: Double = 10.0,
    private val minDb: Double = -55.0, prerollS: Double = 0.25,
) {
    private val frame = (0.025 * SR).toInt()
    private val hop = (STEP * SR).toInt()
    private val gap = (gapS / STEP).roundToInt()
    private val softGap = (softGapS / STEP).roundToInt()
    private val cutGap = (cutGapS / STEP).roundToInt()
    private val softMax = (softMaxS / STEP).roundToInt()
    private val maxLen = (maxTurnS / STEP).roundToInt()
    private val preLen = max(1, (prerollS / STEP).roundToInt())

    private var pending = FloatArray(4 * SR)
    private var pendingLen = 0
    private val levels = DoubleArray(1000)                 // last 10 s of frame levels → noise floor
    private var levelCount = 0
    private var levelNext = 0
    private var floor = -120.0
    private var n = 0                                      // frames seen so far
    private val pre = ArrayDeque<FloatArray>()
    private var hops: MutableList<FloatArray>? = null      // the turn in progress, 10 ms at a time
    private val cuts = mutableListOf<Int>()
    private var startN = 0
    private var lastActive = 0
    private var active = 0

    /** Seconds of speech in the turn in progress (0 when nobody is speaking). */
    val speaking: Double get() = if (hops != null) active * STEP else 0.0

    /** The end of the turn in progress — to show who is speaking before the turn ends. */
    fun recentAudio(seconds: Double = 3.0): FloatArray =
        hops?.let { concat(it.takeLast((seconds / STEP).roundToInt())) } ?: FloatArray(0)

    fun push(block: FloatArray): List<Turn> {
        if (pendingLen + block.size > pending.size) pending = pending.copyOf(max(pending.size * 2, pendingLen + block.size))
        block.copyInto(pending, pendingLen)
        pendingLen += block.size
        val out = mutableListOf<Turn>()
        var pos = 0
        while (pendingLen - pos >= frame) {
            var sum = 0.0
            for (i in pos until pos + frame) sum += pending[i].toDouble() * pending[i].toDouble()
            val db = 10 * log10(sum / frame + 1e-12)
            levels[levelNext] = db
            levelNext = (levelNext + 1) % levels.size
            levelCount = min(levelCount + 1, levels.size)
            if (n % 50 == 0 && levelCount >= 20) {
                val p10 = percentile10()
                floor = if (levelCount >= 300) p10 else min(p10, -50.0)  // a recording may start mid-speech
            }
            val isActive = db > max(floor + marginDb, minDb)
            val hopSamples = pending.copyOfRange(pos, pos + hop)
            val turn = hops
            if (turn == null) {
                pre.addLast(hopSamples)
                if (pre.size > preLen) pre.removeFirst()
                if (isActive) {
                    hops = pre.toMutableList()
                    startN = n - pre.size + 1
                    lastActive = n
                    active = 1
                }
            } else {
                turn.add(hopSamples)
                if (isActive) {
                    val pause = n - lastActive - 1
                    if (pause >= cutGap) cuts.add(turn.size - 1 - (pause + 1) / 2)   // a short pause inside the turn
                    lastActive = n
                    active += 1
                }
                val silent = n - lastActive
                if (silent >= gap || (turn.size >= softMax && silent >= softGap)) out += close(silent)
                else if (turn.size >= maxLen) out += close(0)
            }
            n += 1
            pos += hop
        }
        pending.copyInto(pending, 0, pos, pendingLen)
        pendingLen -= pos
        return out
    }

    /** End of the stream: close the turn in progress. */
    fun flush(): List<Turn> {
        val turn = hops ?: return emptyList()
        turn.add(pending.copyOfRange(0, pendingLen))
        pendingLen = 0
        return close(n - 1 - lastActive)
    }

    private fun close(trailing: Int): List<Turn> {
        val all = hops!!
        val kept = all.subList(0, all.size - max(0, trailing - 10))            // keep 100 ms after the speech
        val turnCuts = cuts.filter { it > 0 && it < kept.size }.map { it * hop }
        val speechS = active * STEP
        val start = startN * hop.toDouble() / SR
        val audio = if (kept.isEmpty()) FloatArray(0) else concat(kept)
        hops = null
        cuts.clear()
        active = 0
        pre.clear()
        if (speechS < minSpeechS || audio.isEmpty()) return emptyList()
        return listOf(Turn(start, start + audio.size.toDouble() / SR, audio, speechS, turnCuts))
    }

    /** numpy.percentile(levels, 10) with linear interpolation. */
    private fun percentile10(): Double {
        val v = levels.copyOf(levelCount).also { it.sort() }
        val p = 0.10 * (v.size - 1)
        val lo = p.toInt()
        val hi = min(lo + 1, v.size - 1)
        return v[lo] + (v[hi] - v[lo]) * (p - lo)
    }
}

internal fun concat(parts: List<FloatArray>): FloatArray {
    val out = FloatArray(parts.sumOf { it.size })
    var at = 0
    for (p in parts) {
        p.copyInto(out, at)
        at += p.size
    }
    return out
}

/**
 * Someone answering quickly leaves several voices in one turn. At each short pause, compare the speech just
 * before it with the speech just after it (up to the neighbouring pauses, at most [windowS], at least
 * [minSideS]) and cut where the voices differ. [embed] gives room-compensated unit embeddings.
 */
fun splitByVoice(
    turn: Turn, embed: (List<FloatArray>) -> List<DoubleArray>, threshold: Double,
    minSideS: Double = 0.6, windowS: Double = 2.0,
): List<Turn> {
    val n = turn.audio.size
    val lo = (minSideS * SR).toInt()
    val win = (windowS * SR).toInt()
    val cuts = turn.cuts.filter { it in lo..(n - lo) }
    if (cuts.isEmpty()) return listOf(turn)
    val bounds = listOf(0) + cuts + listOf(n)
    val windows = mutableListOf<FloatArray>()
    cuts.forEachIndexed { index, c ->
        val k = index + 1
        val a = min(c - lo, max(bounds[k - 1], c - win))
        val b = max(c + lo, min(bounds[k + 1], c + win))
        windows += turn.audio.copyOfRange(a, c)
        windows += turn.audio.copyOfRange(c, b)
    }
    val e = embed(windows)
    val keep = cuts.filterIndexed { i, _ -> dot(e[2 * i], e[2 * i + 1]) < threshold }
    if (keep.isEmpty()) return listOf(turn)
    val edges = listOf(0) + keep + listOf(n)
    val perSample = turn.speechS / max(n, 1)
    return edges.zipWithNext { a, b ->
        Turn(turn.start + a.toDouble() / SR, turn.start + b.toDouble() / SR, turn.audio.copyOfRange(a, b),
            (b - a) * perSample, turn.cuts.filter { it > a && it < b }.map { it - a })
    }
}
