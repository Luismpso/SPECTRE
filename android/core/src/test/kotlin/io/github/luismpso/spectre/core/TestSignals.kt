package io.github.luismpso.spectre.core

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** The generator of scripts/make_android_golden.py: x ← 1664525·x + 1013904223 (mod 2³²). */
class Lcg(seed: Long) {
    private var state = seed and 0xFFFFFFFFL
    fun next(): Double {
        state = (1664525L * state + 1013904223L) and 0xFFFFFFFFL
        return state.toDouble() / 4294967296.0
    }
}

/** synth() of scripts/make_android_golden.py (tone / quiet / talk segments). */
fun synth(spec: JsonArray, seed: Long): FloatArray {
    val rng = Lcg(seed)
    val parts = spec.map { el ->
        val seg = el.jsonObject
        val kind = seg["kind"]!!.jsonPrimitive.content
        val n = (seg["s"]!!.jsonPrimitive.double * SR).toInt()
        val amp = seg["amp"]!!.jsonPrimitive.double
        if (kind == "quiet") FloatArray(n) { (amp * (2 * rng.next() - 1)).toFloat() }
        else {
            val f = seg["f"]!!.jsonPrimitive.double
            FloatArray(n) { i ->
                val t = i.toDouble() / SR
                val x = amp * sin(2 * PI * f * t)
                (if (kind == "talk" && t % 0.26 >= 0.2) 0.0 * x else x).toFloat()
            }
        }
    }
    return concat(parts)
}

fun tone(freq: Double, seconds: Double, amp: Double = 0.1): FloatArray =
    FloatArray((seconds * SR).toInt()) { (amp * sin(2 * PI * freq * (it.toDouble() / SR))).toFloat() }

fun quiet(seconds: Double, rng: Lcg = Lcg(7)): FloatArray =
    FloatArray((seconds * SR).toInt()) { (1e-4 * (2 * rng.next() - 1)).toFloat() }

/** Power of a waveform at one frequency (Goertzel). */
fun power(x: FloatArray, freq: Double): Double {
    val w = 2 * PI * freq / SR
    val c = 2 * cos(w)
    var s1 = 0.0
    var s2 = 0.0
    for (v in x) {
        val s0 = v + c * s1 - s2
        s2 = s1
        s1 = s0
    }
    return s1 * s1 + s2 * s2 - c * s1 * s2
}

/** Fake voices: the energy in three frequency bands is the voice (a mix of voices gives a mixed embedding). */
object BandEmbedder : Embedder {
    val bands = listOf(190.0..230.0, 290.0..330.0, 390.0..430.0)
    override fun embed(wavs: List<FloatArray>): List<FloatArray> = wavs.map { x ->
        val e = bands.map { b -> (b.start.toInt()..b.endInclusive.toInt()).sumOf { power(x, it.toDouble()) } }
        val total = x.sumOf { it.toDouble() * it } * x.size / 2
        val v = (e + maxOf(total - e.sum(), 0.0)).map { sqrt(it) }
        val norm = sqrt(v.sumOf { it * it }) + 1e-12
        FloatArray(v.size) { (v[it] / norm).toFloat() }
    }
}
