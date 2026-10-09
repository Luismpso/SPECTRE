package io.github.luismpso.spectre.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

internal fun dot(a: DoubleArray, b: DoubleArray): Double {
    var s = 0.0
    for (i in a.indices) s += a[i] * b[i]
    return s
}

private fun unit(v: DoubleArray): DoubleArray {
    val norm = sqrt(dot(v, v)) + 1e-12
    return DoubleArray(v.size) { v[it] / norm }
}

/**
 * How turns are given to voices — VoiceRules of spectre.conversation. The values come from voices.json, written by
 * `python -m spectre.export_android` from the calibration of `python -m spectre.conv_eval` for the same model.
 */
data class VoiceRules(
    val threshold: Double = 0.65,       // a turn joins a voice when it is at least this similar to it
    val cut: Double = 0.5,              // a short pause inside a turn is cut when the voices around it are below this
    val mergeMargin: Double = 0.2,      // two voices are merged when their similarity reaches threshold + this
    val followBelowS: Double = 1.0,     // turns with less speech follow the closest voice…
    val newMargin: Double = 0.45,       // …unless they are below threshold − this: clearly someone new
    val minCutSideS: Double = 0.8,      // speech needed on each side of a pause to judge a change of voice
    val room: List<DoubleArray> = emptyList(),   // directions removed from every embedding (room compensation)
    val summary: String = "",           // what the calibration measured
) {
    val merge: Double get() = threshold + mergeMargin
    val newVoice: Double get() = threshold - newMargin

    /** Room compensation: remove the room directions from an embedding and renormalise it. */
    fun project(e: FloatArray): DoubleArray = project(DoubleArray(e.size) { e[it].toDouble() })

    fun project(e: DoubleArray): DoubleArray {
        val v = e.copyOf()
        val along = room.map { dot(e, it) }
        room.forEachIndexed { k, d -> for (i in v.indices) v[i] -= along[k] * d[i] }
        return unit(v)
    }

    companion object {
        /** voices.json (see spectre.export_android). */
        fun fromJson(text: String): VoiceRules {
            val d = Json.parseToJsonElement(text).jsonObject
            fun num(key: String) = d[key]!!.jsonPrimitive.double
            val room = (d["room_directions"] as? JsonArray).orEmpty().map { row ->
                row.jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
            }
            val calibrated = d["calibrated"]?.jsonPrimitive?.boolean ?: false
            val accuracy = d["accuracy"]?.jsonPrimitive?.doubleOrNull
            return VoiceRules(num("threshold"), num("cut"), num("merge_margin"), num("follow_below_s"),
                num("new_margin"), num("min_cut_side_s"), room,
                if (calibrated && accuracy != null)
                    "${(100 * accuracy).roundToInt()} % of the speech to the right person in simulated rooms" else "")
        }

    }
}

/**
 * Online speaker clustering — Voices of spectre.conversation. Turns with enough speech lead: they shape the voices.
 * Short turns follow the closest voice and move if a closer one appears later. Enrolled people are fixed voices.
 */
class Voices(val rules: VoiceRules, enrolled: Map<String, FloatArray> = emptyMap()) {
    class Cluster(var sum: DoubleArray, val embs: MutableList<DoubleArray>, var seconds: Double,
                  val fixed: String?, val label: String)

    val clusters = LinkedHashMap<Int, Cluster>()
    var nextLabel = 1
        private set

    init {
        for ((name, c) in enrolled) clusters[clusters.size] = Cluster(rules.project(c), mutableListOf(), 0.0, name, name)
    }

    fun closest(emb: DoubleArray): Pair<Int?, Double> {
        var best: Int? = null
        var bestSim = -1.0
        for ((cid, c) in clusters) {
            val sim = dot(unit(c.sum), emb)
            if (sim > bestSim) {
                best = cid
                bestSim = sim
            }
        }
        return best to bestSim
    }

    /** (voice id, true if the turn only follows that voice). A new voice when nobody is close enough. */
    fun assign(emb: DoubleArray, seconds: Double): Pair<Int, Boolean> {
        val (closest, sim) = closest(emb)
        if (closest != null && seconds < rules.followBelowS && sim >= rules.newVoice) return closest to true
        var best = closest
        if (best == null || sim < rules.threshold) {
            best = (clusters.keys.maxOrNull() ?: -1) + 1
            clusters[best] = Cluster(DoubleArray(emb.size), mutableListOf(), 0.0, null, "Speaker $nextLabel")
            nextLabel += 1
        }
        val c = clusters.getValue(best)
        if (c.fixed == null) {                                   // enrolled voices keep their reference
            val w = max(seconds, 0.5)
            c.sum = DoubleArray(emb.size) { c.sum[it] + emb[it] * w }
        }
        c.embs.add(emb)
        c.seconds += seconds
        return best to false
    }

    /** Undo an assignment (the turn was only noise); a speaker left without turns disappears. */
    fun unassign(cid: Int, emb: DoubleArray, seconds: Double) {
        val c = clusters[cid] ?: return
        val k = c.embs.indexOfFirst { it === emb }
        if (k < 0) return
        c.embs.removeAt(k)
        c.seconds -= seconds
        if (c.fixed == null) {
            val w = max(seconds, 0.5)
            c.sum = DoubleArray(emb.size) { c.sum[it] - emb[it] * w }
            if (c.embs.isEmpty()) {
                clusters.remove(cid)
                if (c.label == "Speaker ${nextLabel - 1}") nextLabel -= 1
            }
        }
    }

    /** Merge clusters that turned out to be the same voice; returns {removed id: kept id}. */
    fun merge(): Map<Int, Int> {
        var moved = mutableMapOf<Int, Int>()
        var changed = true
        while (changed) {
            changed = false
            val ids = clusters.keys.sorted()
            for (a in ids) for (b in ids) {
                if (a >= b || a !in clusters || b !in clusters) continue
                val ca = clusters.getValue(a)
                val cb = clusters.getValue(b)
                if (ca.fixed != null && cb.fixed != null) continue
                if (dot(unit(ca.sum), unit(cb.sum)) >= rules.merge) {
                    val (keep, drop) = if ((ca.fixed != null || ca.seconds >= cb.seconds) && cb.fixed == null) a to b else b to a
                    val k = clusters.getValue(keep)
                    val d = clusters.remove(drop)!!
                    if (k.fixed == null) k.sum = DoubleArray(k.sum.size) { k.sum[it] + d.sum[it] }
                    k.embs += d.embs
                    k.seconds += d.seconds
                    moved = moved.mapValues { (_, v) -> if (v == drop) keep else v }.toMutableMap()
                    moved[drop] = keep
                    changed = true
                }
            }
        }
        return moved
    }

    fun fixedNames(): Map<Int, String> = clusters.filterValues { it.fixed != null }.mapValues { it.value.fixed!! }

    fun label(cid: Int): String = clusters[cid]?.label ?: "Speaker $cid"
}
