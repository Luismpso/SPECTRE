package io.github.luismpso.spectre.core

import java.util.Locale
import kotlin.math.roundToLong

/** Voice embeddings of 16 kHz waveforms (the SPECTRE model): one unit vector per waveform. */
fun interface Embedder {
    fun embed(wavs: List<FloatArray>): List<FloatArray>
}

/** A line of the conversation: whose voice, when, what was said and the names read in it. */
class Line(
    val start: Double, val end: Double, override var speaker: Int, val seconds: Double,
    val emb: DoubleArray,                 // room-compensated voice embedding
    val raw: FloatArray,                  // as the model gives it (what is saved to recognise someone next time)
    val follows: Boolean,                 // a short turn: it follows the closest voice
) : NamedLine {
    var text: String? = null              // null while being transcribed
    override var names: List<Pair<String, String>>? = null
}

private class Hypothetical(override val speaker: Int) : NamedLine {
    override val names: List<Pair<String, String>>? = null
}

/** What the screen shows for a line. */
data class Caption(val start: Double, val speaker: Int, val label: String, val text: String?)

/** After a change: merge voices that turned out to be one, and let short turns follow the closest voice. */
fun relabel(lines: List<Line>, voices: Voices) {
    val moved = voices.merge()
    for (ln in lines) {
        ln.speaker = moved[ln.speaker] ?: ln.speaker
        if (ln.follows) voices.closest(ln.emb).first?.let { ln.speaker = it }
    }
}

/**
 * The lines, the voices and the current names (Conversation in spectre.conversation). A turn goes through
 * [hear] (its voice: the line appears, already named if the rules allow) and [transcribed] (its words and the
 * names in them: names may appear, also on earlier lines). Safe to call from a worker while the screen reads.
 */
class Conversation(
    private val embedder: Embedder,
    val voices: Voices,
    private val readNames: (String) -> List<Pair<String, String>> = ::ruleNames,
) {
    private val lines = mutableListOf<Line>()
    private var names: Map<Int, Naming> = resolveNames(emptyList(), voices.fixedNames())

    /** (voice, label) of the turn in progress, before it ends; null when nobody is speaking. */
    @Volatile
    var speaking: Pair<Int, String>? = null

    fun embed(wavs: List<FloatArray>): List<DoubleArray> = embedder.embed(wavs).map { voices.rules.project(it) }

    /** Cut a turn where the voice changes (someone answered quickly). */
    fun split(turn: Turn): List<Turn> = splitByVoice(turn, ::embed, voices.rules.cut, voices.rules.minCutSideS)

    /** The voice of a finished turn; its line appears before the words. */
    fun hear(turn: Turn): Line {
        val raw = embedder.embed(listOf(turn.audio))[0]
        val emb = voices.rules.project(raw)
        synchronized(this) {
            val (spk, follows) = voices.assign(emb, turn.speechS)
            val line = Line(turn.start, turn.end, spk, turn.speechS, emb, raw, follows)
            lines += line
            relabel(lines, voices)
            update()
            return line
        }
    }

    /** The words of a line, and the names in them. False, and the line disappears, when it was only noise. */
    fun transcribed(line: Line, text: String): Boolean {
        val clean = cleanTranscript(text)
        val found = if (clean.isEmpty()) emptyList() else checkedNames(readNames(clean), clean)
        synchronized(this) {
            if (line !in lines) return false
            if (clean.isEmpty()) {
                lines.remove(line)
                if (!line.follows) voices.unassign(line.speaker, line.emb, line.seconds)
                relabel(lines, voices)
            } else {
                line.text = clean
                line.names = found
            }
            update()
            return clean.isNotEmpty()
        }
    }

    /** Who is speaking in the turn in progress, with the same rules as for a finished line: right after
     *  "Olá João", a voice never heard before is shown as João at once. */
    fun peek(audio: FloatArray) {
        val emb = voices.rules.project(embedder.embed(listOf(audio))[0])
        synchronized(this) {
            val (closest, sim) = voices.closest(emb)
            val cid = if (closest == null || sim < voices.rules.threshold) (voices.clusters.keys.maxOrNull() ?: -1) + 1 else closest
            val probe = resolveNames(lines + Hypothetical(cid), voices.fixedNames())
            speaking = cid to (probe[cid]?.name ?: voices.clusters[cid]?.label ?: "a new voice")
        }
    }

    fun label(spk: Int): String = synchronized(this) { names[spk]?.name ?: voices.label(spk) }

    fun captions(): List<Caption> = synchronized(this) {
        lines.map { Caption(it.start, it.speaker, names[it.speaker]?.name ?: voices.label(it.speaker), it.text) }
    }

    /** The names of the voices heard, and why. */
    fun namings(): Map<Int, Naming> = synchronized(this) {
        val heard = lines.map { it.speaker }.toSet()
        names.filterKeys { it in heard }
    }

    /** Raw embeddings and seconds of speech of each named voice (to recognise those people next time). */
    fun namedVoices(): Map<String, Pair<List<FloatArray>, Double>> = synchronized(this) {
        names.filter { (spk, _) -> voices.clusters[spk]?.fixed == null }.mapNotNull { (spk, n) ->
            val own = lines.filter { it.speaker == spk }
            if (own.isEmpty()) null else n.name to (own.map { it.raw } to own.sumOf { it.seconds })
        }.toMap()
    }

    fun transcript(): String = synchronized(this) {
        val out = lines.filter { !it.text.isNullOrEmpty() }.map { "[${clock(it.start)}] ${label(it.speaker)}: ${it.text}" }
            .toMutableList()
        val found = namings()
        if (found.isNotEmpty()) {
            out += "\nHow the names were found:"
            found.values.sortedBy { it.name }.forEach { out += "  ${it.name}: " + it.evidence.take(3).joinToString("; ") }
        }
        out.joinToString("\n")
    }

    private fun update() {
        names = resolveNames(lines, voices.fixedNames())
    }
}

/** mm:ss.s */
fun clock(t: Double): String {
    val tenths = (t * 10).roundToLong()
    return String.format(Locale.ROOT, "%02d:%04.1f", tenths / 600, (tenths % 600) / 10.0)
}
