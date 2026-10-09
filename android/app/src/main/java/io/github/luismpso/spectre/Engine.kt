package io.github.luismpso.spectre

import android.content.Context
import android.os.Build
import io.github.luismpso.spectre.core.Caption
import io.github.luismpso.spectre.core.Conversation
import io.github.luismpso.spectre.core.SR
import io.github.luismpso.spectre.core.Turn
import io.github.luismpso.spectre.core.TurnSegmenter
import io.github.luismpso.spectre.core.VoiceRules
import io.github.luismpso.spectre.core.Voices
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Live captions, all on the phone: microphone → turns (energy VAD) → voices (the SPECTRE model on ONNX Runtime,
 * room-compensated) → words (Whisper on whisper.cpp) → names (rules). The screen only reads [state], so another
 * display (smart glasses) can show the same captions.
 *
 * Threads: the microphone delivers 100 ms blocks; "turns" cuts them into turns and says who is speaking before a
 * turn ends; "work" gives each finished turn its voice, then its words — in order, one turn at a time.
 */
class Engine(private val context: Context) : AutoCloseable {

    data class State(
        val captions: List<Caption> = emptyList(),
        val speaking: Pair<Int, String>? = null,     // the turn in progress: (voice, name or label)
        val listening: Boolean = false,
        val busy: String? = null,                    // loading or downloading, shown instead of the status
        val progress: Float? = null,                 // download progress, 0..1
        val error: String? = null,
        val waiting: Int = 0,                        // finished turns not transcribed yet
        val model: WhisperModel = WhisperModel.SMALL,
        val downloaded: Set<WhisperModel> = emptySet(),
        val calibration: String = "",                // what the voice calibration measured (voices.json)
    )

    private val _state = MutableStateFlow(State(downloaded = downloadedModels()))
    val state: StateFlow<State> = _state

    private val turnsThread = Executors.newSingleThreadExecutor { Thread(it, "spectre-turns") }
    private val workThread = Executors.newSingleThreadExecutor { Thread(it, "spectre-work") }

    // models: loaded once, used from the work thread (Whisper is not thread-safe; ONNX Runtime is)
    private var embedder: OnnxEmbedder? = null
    private var rules: VoiceRules? = null
    private var whisper: Whisper? = null
    private var whisperModel: WhisperModel? = null

    // the conversation: kept across Stop/Start until cleared
    @Volatile private var conversation: Conversation? = null
    private var segmenter: TurnSegmenter? = null              // only touched on the turns thread
    private var capture: AudioCapture? = null
    private var heard = 0L                                    // samples since the conversation started
    private var lastPeek = 0L
    private val waiting = AtomicInteger(0)
    @Volatile private var generation = 0                      // clearing drops the work of the old conversation

    fun choose(model: WhisperModel) {
        if (!_state.value.listening && _state.value.busy == null) _state.update { it.copy(model = model, error = null) }
    }

    /** Load the voice model and the chosen Whisper model (downloading it the first time). */
    suspend fun prepare() = withContext(Dispatchers.IO) {
        val model = _state.value.model
        _state.update { it.copy(error = null) }
        try {
            checkProcessor()
            if (embedder == null) {
                _state.update { it.copy(busy = "Loading the voice model…") }
                val r = VoiceRules.fromJson(context.assets.open("voices.json").use { String(it.readBytes()) })
                embedder = OnnxEmbedder(context.assets.open("spectre_ecapa.onnx").use { it.readBytes() })
                rules = r
                _state.update { it.copy(calibration = r.summary) }
            }
            if (whisperModel != model) {
                if (!ModelStore.isReady(context, model)) {
                    _state.update { it.copy(busy = "Downloading Whisper ${model.label} (${model.megabytes} MB)…", progress = 0f) }
                    ModelStore.download(context, model) { p -> _state.update { it.copy(progress = p) } }
                    _state.update { it.copy(progress = null, downloaded = downloadedModels()) }
                }
                _state.update { it.copy(busy = "Loading Whisper ${model.label}…") }
                val path = ModelStore.file(context, model).path
                workThread.submit {                                   // never swap it under a transcription
                    whisper?.close()
                    whisper = null
                    whisper = Whisper(path)
                    whisperModel = model
                }.get()
            }
            _state.update { it.copy(busy = null) }
            true
        } catch (e: Exception) {
            val why = (e.cause ?: e).message ?: e.javaClass.simpleName
            _state.update { it.copy(busy = null, progress = null, error = why) }
            false
        }
    }

    /** Start listening (after [prepare]); a stopped conversation goes on where it was. */
    fun start() {
        val emb = embedder ?: return
        val r = rules ?: return
        if (_state.value.listening) return
        if (conversation == null) {
            conversation = Conversation(emb, Voices(r))
            turnsThread.execute {
                segmenter = TurnSegmenter()
                heard = 0
                lastPeek = 0
            }
        }
        val gen = generation
        val mic = AudioCapture({ block -> turnsThread.execute { onBlock(block, gen) } }) { problem ->
            context.mainExecutor.execute {                       // the microphone stopped (another app took it…)
                if (capture != null) stop()
                _state.update { it.copy(error = problem) }
            }
        }
        try {
            mic.start()
        } catch (e: Exception) {
            _state.update { it.copy(error = e.message ?: "The microphone is not available") }
            return
        }
        capture = mic
        _state.update { it.copy(listening = true, error = null) }
    }

    /** Stop listening: the turn in progress ends and the turns still waiting are transcribed. */
    fun stop() {
        capture?.stop()
        capture = null
        val gen = generation
        turnsThread.execute {
            if (gen != generation) return@execute
            segmenter?.flush()?.forEach { queue(it, gen) }
            conversation?.speaking = null
            publish()
        }
        _state.update { it.copy(listening = false) }
    }

    /** Forget the conversation (when stopped): voices, names and captions. */
    fun clear() {
        if (_state.value.listening) return
        generation += 1
        conversation = null
        turnsThread.execute { segmenter = null }
        _state.update { it.copy(captions = emptyList(), speaking = null, error = null) }
    }

    /** The conversation as text, with how the names were found. */
    fun transcript(): String = conversation?.transcript().orEmpty()

    private fun onBlock(block: FloatArray, gen: Int) {
        val seg = segmenter ?: return
        val conv = conversation ?: return
        if (gen != generation) return
        heard += block.size
        try {
            val turns = seg.push(block)
            if (turns.isNotEmpty()) {
                conv.speaking = null
                turns.forEach { queue(it, gen) }
                publish()
            } else if (seg.speaking >= 1.0 && heard - lastPeek >= SR / 2) {
                lastPeek = heard                                 // whose voice is this, before the turn ends
                conv.peek(seg.recentAudio())
                publish()
            }
        } catch (e: Exception) {
            fail(e)
        }
    }

    private fun queue(turn: Turn, gen: Int) {
        waiting.incrementAndGet()
        workThread.execute {
            try {
                val conv = conversation
                val w = whisper
                if (gen == generation && conv != null && w != null) {
                    for (part in conv.split(turn)) {             // someone answered quickly: one line each
                        val line = conv.hear(part)               // the line appears with its voice…
                        publish()
                        conv.transcribed(line, w.transcribe(part.audio))   // …then its words and names
                        if (gen != generation) break
                    }
                }
            } catch (e: Exception) {
                fail(e)                                          // one bad turn never stops the captions
            } finally {
                waiting.decrementAndGet()
                publish()
            }
        }
    }

    private fun publish() {
        val conv = conversation ?: return
        val captions = conv.captions()
        _state.update { it.copy(captions = captions, speaking = conv.speaking, waiting = waiting.get()) }
    }

    private fun fail(e: Exception) {
        _state.update { it.copy(error = "${e.javaClass.simpleName}: ${e.message}") }
    }

    private fun downloadedModels() = WhisperModel.entries.filter { ModelStore.isReady(context, it) }.toSet()

    /** The native code is built for ARMv8.2 with dot product and half floats (phones from 2018 on). */
    private fun checkProcessor() {
        if (Build.SUPPORTED_ABIS.firstOrNull() != "arm64-v8a") return
        val features = runCatching { File("/proc/cpuinfo").readLines() }.getOrDefault(emptyList())
            .firstOrNull { it.startsWith("Features") }?.substringAfter(':')?.split(' ')?.toSet() ?: return
        check("asimddp" in features && "fphp" in features) {
            "This phone's processor is too old for SPECTRE (it needs ARMv8.2 with dot product instructions)"
        }
    }

    override fun close() {
        capture?.stop()
        capture = null
        generation += 1
        turnsThread.shutdownNow()
        workThread.execute {
            whisper?.close()
            embedder?.close()
        }
        workThread.shutdown()
        workThread.awaitTermination(2, TimeUnit.SECONDS)
    }
}
