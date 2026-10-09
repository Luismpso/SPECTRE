package io.github.luismpso.spectre

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import io.github.luismpso.spectre.core.SR

/** The microphone at 16 kHz mono, delivered in blocks of 100 ms on a background thread. */
class AudioCapture(private val onBlock: (FloatArray) -> Unit, private val onError: (String) -> Unit = {}) {
    @Volatile
    private var running = false
    private var thread: Thread? = null

    /** Throws when the microphone cannot be opened (in use by another app, for example). */
    @SuppressLint("MissingPermission")    // asked for before start() is called
    fun start() {
        val minBuffer = AudioRecord.getMinBufferSize(SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        // VOICE_RECOGNITION: no automatic gain or noise suppression on most phones, which keeps voices as they are
        val record = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, SR, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuffer, 2 * SR))
        try {
            check(record.state == AudioRecord.STATE_INITIALIZED) { "The microphone is not available" }
            record.startRecording()
            check(record.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "The microphone is being used by another app" }
        } catch (e: Exception) {
            record.release()
            throw e
        }
        running = true
        thread = Thread({
            val buffer = ShortArray(SR / 10)
            try {
                while (running) {
                    var got = 0
                    while (running && got < buffer.size) {
                        val read = record.read(buffer, got, buffer.size - got)
                        check(read >= 0) { "The microphone stopped (error $read)" }
                        if (read == 0) break
                        got += read
                    }
                    if (got > 0) onBlock(FloatArray(got) { buffer[it] / 32768f })
                }
            } catch (e: Exception) {
                if (running) onError(e.message ?: "The microphone stopped")
                running = false
            } finally {
                runCatching { record.stop() }
                record.release()
            }
        }, "spectre-microphone").apply { start() }
    }

    fun stop() {
        running = false
        thread?.let { if (it !== Thread.currentThread()) it.join(1_000) }
        thread = null
    }
}
