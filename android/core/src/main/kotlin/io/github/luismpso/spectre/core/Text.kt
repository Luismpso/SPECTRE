package io.github.luismpso.spectre.core

import java.text.Normalizer

/** Lower-case, accent-free form used to compare names ("João" == "joao"). Same as fold() in spectre.conversation. */
fun fold(text: String): String =
    Normalizer.normalize(text, Normalizer.Form.NFKD).filter { it.code < 128 }.lowercase().trim()

/** Letter runs of a folded (ASCII) text. */
internal fun asciiWords(folded: String): List<String> = Regex("[a-z]+").findAll(folded).map { it.value }.toList()

/** Python's difflib.SequenceMatcher(None, a, b).ratio(): 2·matches / (|a| + |b|), with the same matching blocks. */
fun ratio(a: String, b: String): Double {
    if (a.isEmpty() && b.isEmpty()) return 1.0
    val positions = HashMap<Char, MutableList<Int>>()
    b.forEachIndexed { j, ch -> positions.getOrPut(ch) { mutableListOf() }.add(j) }

    fun longest(alo: Int, ahi: Int, blo: Int, bhi: Int): Triple<Int, Int, Int> {
        var bestI = alo
        var bestJ = blo
        var bestSize = 0
        var lengths = HashMap<Int, Int>()
        for (i in alo until ahi) {
            val next = HashMap<Int, Int>()
            for (j in positions[a[i]] ?: emptyList()) {
                if (j < blo) continue
                if (j >= bhi) break
                val k = (lengths[j - 1] ?: 0) + 1
                next[j] = k
                if (k > bestSize) {
                    bestI = i - k + 1
                    bestJ = j - k + 1
                    bestSize = k
                }
            }
            lengths = next
        }
        return Triple(bestI, bestJ, bestSize)
    }

    var matches = 0
    val queue = ArrayDeque(listOf(intArrayOf(0, a.length, 0, b.length)))
    while (queue.isNotEmpty()) {
        val (alo, ahi, blo, bhi) = queue.removeLast()
        val (i, j, k) = longest(alo, ahi, blo, bhi)
        if (k > 0) {
            matches += k
            if (alo < i && blo < j) queue.addLast(intArrayOf(alo, i, blo, j))
            if (i + k < ahi && j + k < bhi) queue.addLast(intArrayOf(i + k, ahi, j + k, bhi))
        }
    }
    return 2.0 * matches / (a.length + b.length)
}

/** Whisper's usual inventions on silence or noise, and annotations such as "[BLANK_AUDIO]" or "(música)". */
private val HALLUCINATIONS = listOf("amara.org", "obrigado por assistir", "inscreva-se", "subscreva o canal",
    "legendas pela comunidade")

/** The transcript without annotations; empty when it is only noise or a typical hallucination. */
fun cleanTranscript(text: String): String {
    val t = text.replace(Regex("\\[[^\\]]*]|\\([^)]*\\)|\\*[^*]*\\*"), " ").replace(Regex("\\s+"), " ").trim()
    return if (t.isEmpty() || HALLUCINATIONS.any { it in t.lowercase() }) "" else withoutLoops(t)
}

private val SENTENCES = Regex("[^.!?…]+[.!?…]*")
private val REPEATED_WORD = Regex("(?<!\\p{L})(\\p{L}+)(?:[\\s,]+\\1(?!\\p{L})){3,}", RegexOption.IGNORE_CASE)  // no \b: it differs on Android

/**
 * Whisper sometimes loops until it runs out of tokens ("Então vamos andando. Então vamos andando. Então va"): a
 * sentence that only repeats the one before (two words or more), or part of it (three or more), is dropped; so is a
 * word said four times running.
 */
private fun withoutLoops(text: String): String {
    val kept = mutableListOf<String>()
    var last = ""
    for (m in SENTENCES.findAll(REPEATED_WORD.replace(text, "$1"))) {
        val sentence = m.value.trim()
        val key = fold(sentence).replace(Regex("[^a-z0-9 ]"), "").replace(Regex("\\s+"), " ").trim()
        val spaces = key.count { it == ' ' }
        if (key.isEmpty() || (spaces >= 1 && key == last) || (spaces >= 2 && " $key " in " $last ")) continue
        kept += sentence
        last = key
    }
    return kept.joinToString(" ")
}
