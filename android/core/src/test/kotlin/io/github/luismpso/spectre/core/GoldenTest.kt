package io.github.luismpso.spectre.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs

/** The same inputs as scripts/make_android_golden.py gave the Python implementation: the same outputs here. */
class GoldenTest {
    private val g: JsonObject = Json.parseToJsonElement(
        javaClass.getResource("/golden.json")!!.readText()).jsonObject

    private fun pairs(a: JsonArray) = a.map { it.jsonArray[0].jsonPrimitive.content to it.jsonArray[1].jsonPrimitive.content }

    @Test fun foldAndRatio() {
        for (c in g["fold"]!!.jsonArray) assertEquals(c.jsonArray[1].jsonPrimitive.content, fold(c.jsonArray[0].jsonPrimitive.content))
        for (c in g["ratio"]!!.jsonArray) {
            val (a, b, r) = c.jsonArray
            assertEquals("${a.jsonPrimitive.content}/${b.jsonPrimitive.content}", r.jsonPrimitive.double,
                ratio(a.jsonPrimitive.content, b.jsonPrimitive.content), 1e-12)
        }
    }

    @Test fun namesInLines() {
        for (case in g["lines"]!!.jsonArray) {
            val o = case.jsonObject
            val line = o["line"]!!.jsonPrimitive.content
            val found = ruleNames(line)
            assertEquals(line, pairs(o["rule_names"]!!.jsonArray), found)
            assertEquals(line, pairs(o["checked"]!!.jsonArray), checkedNames(found, line))
            for ((name, v) in o["vocative"]!!.jsonObject) assertEquals("$line / $name", v.jsonPrimitive.contentOrNull, vocative(name, line))
            for ((name, own) in o["own"]!!.jsonObject) assertEquals("$line / $name", own.jsonPrimitive.boolean, saysOwnName(name, line))
        }
        for (c in g["written_in"]!!.jsonArray) {
            val (n, l, r) = c.jsonArray
            assertEquals(r.jsonPrimitive.boolean, writtenIn(n.jsonPrimitive.content, l.jsonPrimitive.content))
        }
        for (c in g["check_role"]!!.jsonArray) {
            val (n, r, l, out) = c.jsonArray
            assertEquals(out.jsonPrimitive.contentOrNull, checkRole(n.jsonPrimitive.content, r.jsonPrimitive.content, l.jsonPrimitive.content))
        }
    }

    private class Scripted(override val speaker: Int, override val names: List<Pair<String, String>>?) : NamedLine

    @Test fun whoIsWho() {
        for (case in g["resolve"]!!.jsonArray) {
            val o = case.jsonObject
            val lines = o["lines"]!!.jsonArray.map { l ->
                val (spk, names) = l.jsonArray
                Scripted(spk.jsonPrimitive.int, if (names is JsonNull) null else pairs(names.jsonArray))
            }
            val fixed = o["fixed"]!!.jsonObject.mapKeys { it.key.toInt() }.mapValues { it.value.jsonPrimitive.content }
            val got = resolveNames(lines, fixed)
            val want = o["result"]!!.jsonObject
            assertEquals(want.keys.map(String::toInt).toSet(), got.keys)
            for ((spk, w) in want) {
                val n = got.getValue(spk.toInt())
                val wo = w.jsonObject
                assertEquals(wo["name"]!!.jsonPrimitive.content, n.name)
                assertEquals(wo["score"]!!.jsonPrimitive.doubleOrNull ?: Double.POSITIVE_INFINITY, n.score, 1e-9)
                assertEquals(wo["evidence"]!!.jsonArray.map { it.jsonPrimitive.content }, n.evidence)
            }
        }
    }

    @Test fun turns() {
        val s = g["segmenter"]!!.jsonObject
        val wav = synth(s["spec"]!!.jsonArray, s["seed"]!!.jsonPrimitive.int.toLong())
        assertEquals(s["samples"]!!.jsonPrimitive.int, wav.size)
        val seg = TurnSegmenter()
        val turns = (0 until wav.size step 1600).flatMap { seg.push(wav.copyOfRange(it, minOf(it + 1600, wav.size))) } + seg.flush()
        val want = s["turns"]!!.jsonArray
        assertEquals(want.size, turns.size)
        want.zip(turns).forEach { (w, t) ->
            val o = w.jsonObject
            assertEquals(o["start"]!!.jsonPrimitive.double, t.start, 1e-9)
            assertEquals(o["end"]!!.jsonPrimitive.double, t.end, 1e-9)
            assertEquals(o["speech_s"]!!.jsonPrimitive.double, t.speechS, 1e-9)
            assertEquals(o["cuts"]!!.jsonArray.map { it.jsonPrimitive.int }, t.cuts)
            assertEquals(o["samples"]!!.jsonPrimitive.int, t.audio.size)
        }
    }

    @Test fun voices() {
        for (case in g["voices"]!!.jsonArray) {
            val o = case.jsonObject
            val rules = VoiceRules.fromJson(o["rules"].toString())
            val voices = Voices(rules)
            val seconds = o["seconds"]!!.jsonArray.map { it.jsonPrimitive.double }
            val lines = mutableListOf<Line>()
            o["raw"]!!.jsonArray.forEachIndexed { k, r ->
                val raw = r.jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
                val e = rules.project(raw)
                val (spk, follows) = voices.assign(e, seconds[k])
                lines += Line(k.toDouble(), k + 1.0, spk, seconds[k], e, FloatArray(0), follows)
                relabel(lines, voices)
                val step = o["steps"]!!.jsonArray[k].jsonObject
                assertEquals("step $k", step["assigned"]!!.jsonPrimitive.int, spk)
                assertEquals("step $k", step["follows"]!!.jsonPrimitive.boolean, follows)
                assertEquals("step $k", step["labels"]!!.jsonArray.map { it.jsonPrimitive.int }, lines.map { it.speaker })
            }
        }
    }

    /** The exported model (python -m spectre.export_android) gives PyTorch's embeddings through ONNX Runtime. */
    @Test fun onnxModel() {
        val o = g["onnx"]?.jsonObject
        val assets = File(System.getProperty("spectre.assets") ?: "../app/src/main/assets")
        val model = File(assets, "spectre_ecapa.onnx")
        val rules = File(assets, "voices.json")
        assumeTrue("no exported model", o != null && model.exists() && rules.exists())
        val exported = Json.parseToJsonElement(rules.readText()).jsonObject["model"]!!.jsonPrimitive.content
        assumeTrue("fixtures made with another model", exported == o!!["model"]!!.jsonPrimitive.content)
        val env = OrtEnvironment.getEnvironment()
        env.createSession(model.readBytes()).use { session ->
            o["clips"]!!.jsonArray.forEachIndexed { k, spec ->
                val wav = synth(spec.jsonArray, 5)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(wav), longArrayOf(1, wav.size.toLong())).use { input ->
                    session.run(mapOf("wav" to input)).use { out ->
                        @Suppress("UNCHECKED_CAST")
                        val emb = (out[0].value as Array<FloatArray>)[0]
                        val want = o["embeddings"]!!.jsonArray[k].jsonArray.map { it.jsonPrimitive.double }
                        val worst = want.indices.maxOf { abs(want[it] - emb[it]) }
                        assert(worst < 1e-3) { "clip $k differs by $worst" }
                    }
                }
            }
        }
    }
}
