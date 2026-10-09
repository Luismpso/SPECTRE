package io.github.luismpso.spectre.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The conversation tests of tests/test_conversation.py, on the Kotlin port (no models: tones are voices). */
class BehaviourTest {
    // Each line is a tone: the band says whose voice it is, the exact frequency which line.
    private val script = listOf(
        Triple("Joana", 200.0, "Olá João! Há quanto tempo não te via."),
        Triple("João", 300.0, "Olá Joana, tudo bem contigo?"),
        Triple("Joana", 205.0, "Tudo ótimo. Olha, apresento-te o Pedro."),
        Triple("Pedro", 400.0, "Muito prazer, João. Eu sou o Pedro."),
        Triple("João", 305.0, "O prazer é meu, Pedro. A Rita também vem jantar?"),
        Triple("Joana", 210.0, "A Rita chega mais tarde."),
        Triple("Pedro", 405.0, "Então vamos andando."),
    )
    private val who = script.map { it.first }
    private val rules = VoiceRules(threshold = 0.6, cut = 0.6, mergeMargin = 0.2, followBelowS = 1.0, newMargin = 0.3,
        minCutSideS = 0.6)

    private fun transcribe(audio: FloatArray): String {
        val (_, f, text) = script.maxBy { power(audio, it.second) }
        return if (power(audio, f) > 1.0) text else ""
    }

    private fun dialogue(gap: Double = 1.0): FloatArray {
        val rng = Lcg(3)
        return concat(listOf(quiet(1.0, rng)) + script.flatMap { listOf(tone(it.second, 1.6), quiet(gap, rng)) })
    }

    private fun turnsOf(wav: FloatArray): List<Turn> {
        val seg = TurnSegmenter()
        return (wav.indices step 1600).flatMap { seg.push(wav.copyOfRange(it, minOf(it + 1600, wav.size))) } + seg.flush()
    }

    private fun Conversation.process(turns: List<Turn>, before: (Conversation) -> Unit = {}) {
        for (turn in turns) for (part in split(turn)) {
            val line = hear(part)
            before(this)
            transcribed(line, transcribe(part.audio))
        }
    }

    private fun conversation(enrolled: Map<String, FloatArray> = emptyMap()) =
        Conversation(BandEmbedder, Voices(rules, enrolled))

    @Test fun peopleAreNamedFromWhatTheySay() {
        val conv = conversation()
        val seen = mutableListOf<String>()
        conv.process(turnsOf(dialogue())) { seen += it.captions().last().label }
        val captions = conv.captions()
        assertEquals(script.map { it.third }, captions.map { it.text })
        assertEquals(who, captions.map { it.label })
        assertEquals(listOf("Speaker 1", "João"), seen.take(2))              // João's name shows before his words do
        assertTrue("Rita" !in captions.map { it.label })
        val transcript = conv.transcript()
        assertTrue(transcript, transcript.startsWith("[00:00.7] Joana: Olá João!") && "How the names were found:" in transcript)
    }

    @Test fun quickRepliesAreCutWhereTheVoiceChanges() {
        val conv = conversation()
        conv.process(turnsOf(dialogue(gap = 0.3)))                          // answers 0.3 s after the other person
        assertEquals(who, conv.captions().map { it.label })
    }

    @Test fun noiseIsDroppedAndForgotten() {
        val conv = conversation()
        val line = conv.hear(Turn(0.0, 1.0, tone(600.0, 1.0), 1.0))       // a door closing: no words
        assertFalse(conv.transcribed(line, ""))
        assertTrue(conv.captions().isEmpty() && conv.voices.clusters.isEmpty())
        assertEquals(1, conv.voices.nextLabel)
    }

    @Test fun theVoiceOfTheTurnInProgress() {
        val conv = conversation()
        conv.process(turnsOf(dialogue()).take(1))                             // Joana: "Olá João! …"
        conv.peek(tone(302.0, 1.5))                                            // a voice never heard starts talking
        assertEquals(1 to "João", conv.speaking)
        conv.process(turnsOf(dialogue()).drop(1).take(1))
        conv.peek(tone(205.0, 1.5))
        assertEquals(0 to "Joana", conv.speaking)
    }

    @Test fun enrolledPeopleAreRecognisedByVoice() {
        val conv = conversation(mapOf("João" to floatArrayOf(0f, 1f, 0f, 0f)))
        conv.process(turnsOf(dialogue()))
        assertEquals(who, conv.captions().map { it.label })
        assertEquals(listOf("recognised by voice (enrolled)"), conv.namings().getValue(0).evidence)
        assertEquals(setOf("Joana", "Pedro"), conv.namedVoices().keys)          // what can be remembered
    }

    @Test fun whisperAnnotationsAndTimestamps() {
        assertEquals("", cleanTranscript(" [BLANK_AUDIO] "))
        assertEquals("Olá Pedro.", cleanTranscript("(música) Olá Pedro."))
        assertEquals("", cleanTranscript("Obrigado por assistir!"))
        assertEquals("Hoje eu, a quente pimp não te via.",                       // whisper.cpp loops, as heard
            cleanTranscript("Hoje eu, a quente pimp não te via. Hoje eu, a quente pimp não te via. Hoje eu, a quente pimp não te"))
        assertEquals("Tudo o ópio, olha, acusando tu peço, trabalha comigo.",
            cleanTranscript("Tudo o ópio, olha, acusando tu peço, trabalha comigo. Olha, acusando tu peço, trabalha comigo. Olha, acusando tu peço, trabalha"))
        assertEquals("Hoje mem. Você.", cleanTranscript("Hoje mem mem mem mem mem. Você você, você você."))
        assertEquals("Então vamos.", cleanTranscript("Então vamos. Então vamos. Então vamos."))
        assertEquals("Sim, eu sei. Sim. Olá. Olá.", cleanTranscript("Sim, eu sei. Sim. Olá. Olá."))   // real speech stays
        assertEquals("00:03.8", clock(3.84))
        assertEquals("01:01.3", clock(61.26))
    }

    @Test fun voicesJsonFromTheExport() {
        val rules = VoiceRules.fromJson("""{"model": "run_x/best.pt", "calibrated": true, "threshold": 0.4, "cut": 0.3,
            "merge_margin": 0.2, "follow_below_s": 1.0, "new_margin": 0.3, "min_cut_side_s": 0.8,
            "room_directions": [[0, 0, 0, 1.0]], "accuracy": 0.891}""")
        assertEquals(0.6, rules.merge, 1e-12)
        assertEquals(0.1, rules.newVoice, 1e-12)
        assertEquals("89 % of the speech to the right person in simulated rooms", rules.summary)
        val p = rules.project(floatArrayOf(1f, 0f, 0f, 2f))
        assertEquals(listOf(1.0, 0.0, 0.0, 0.0), p.map { Math.round(it * 1e9) / 1e9 })   // the room is removed
    }
}
