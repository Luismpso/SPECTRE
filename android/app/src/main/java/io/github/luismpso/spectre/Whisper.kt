package io.github.luismpso.spectre

import io.github.luismpso.spectre.core.SR

/** whisper.cpp through JNI (src/main/cpp/whisper_jni.cpp). */
internal object WhisperNative {
    init {
        System.loadLibrary("spectre_whisper")
    }

    external fun init(modelPath: String): Long
    external fun transcribe(ctx: Long, audio: FloatArray, language: String, threads: Int, audioCtx: Int,
                           maxTokens: Int): ByteArray
    external fun free(ctx: Long)
    external fun systemInfo(): String
}

private const val MIN_AUDIO_CTX = 512     // ≈ 10 s

/** A Whisper model in memory. Not thread-safe: use it from one thread. */
class Whisper(modelPath: String, private val language: String = "pt") : AutoCloseable {
    private var ctx = WhisperNative.init(modelPath).also { require(it != 0L) { "Could not load the Whisper model" } }

    // the big cores of a phone: 4 threads is the sweet spot for whisper.cpp on 8-core chips
    private val threads = Runtime.getRuntime().availableProcessors().let { if (it >= 6) 4 else maxOf(1, it - 1) }

    /**
     * 16 kHz mono audio → text. Whisper encodes 30 s windows; a turn lasts a few seconds, so only the turn is
     * encoded (audio_ctx: 50 frames per second, plus a margin), but never less than ~10 s — shorter contexts
     * make Whisper loop or invent words. About 3× faster than 30 s windows, about as accurate (test conversation).
     */
    fun transcribe(audio: FloatArray): String {
        val input = if (audio.size < 17_600) audio.copyOf(17_600) else audio     // at least 1.1 s
        val audioCtx = minOf(1500, maxOf(MIN_AUDIO_CTX, input.size / 320 + 64))
        val maxTokens = (10 * input.size / SR) + 10                              // speech is ~5 tokens a second
        return String(WhisperNative.transcribe(ctx, input, language, threads, audioCtx, maxTokens), Charsets.UTF_8).trim()
    }

    override fun close() {
        if (ctx != 0L) WhisperNative.free(ctx)
        ctx = 0L
    }
}
