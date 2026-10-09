package io.github.luismpso.spectre

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/** Quantised Whisper models for whisper.cpp, downloaded from Hugging Face the first time they are used. */
enum class WhisperModel(val label: String, val file: String, val bytes: Long, val note: String) {
    BASE("base", "ggml-base-q5_1.bin", 59_707_625, "fastest"),
    SMALL("small", "ggml-small-q5_1.bin", 190_085_487, "balanced"),
    TURBO("turbo", "ggml-large-v3-turbo-q5_0.bin", 574_041_195, "best, slowest");

    val url: String get() = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$file"
    val megabytes: Long get() = (bytes + 500_000) / 1_000_000
}

object ModelStore {
    fun file(context: Context, model: WhisperModel) = File(context.filesDir, "models/${model.file}")

    fun isReady(context: Context, model: WhisperModel) = file(context, model).let { it.exists() && it.length() == model.bytes }

    /** Download a model (resuming an interrupted download); [onProgress] gets 0..1. */
    fun download(context: Context, model: WhisperModel, onProgress: (Float) -> Unit): File {
        val dest = file(context, model)
        if (isReady(context, model)) return dest
        dest.parentFile!!.mkdirs()
        val part = File(dest.path + ".part")
        if (part.length() > model.bytes) part.delete()
        if (part.length() < model.bytes) fetch(model, part, onProgress)
        check(part.length() == model.bytes) { "The download was interrupted; try again" }
        check(part.renameTo(dest)) { "Could not save the model" }
        return dest
    }

    private fun fetch(model: WhisperModel, part: File, onProgress: (Float) -> Unit) {
        val have = part.length()
        val connection = (URL(model.url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 30_000
            instanceFollowRedirects = true
            if (have > 0) setRequestProperty("Range", "bytes=$have-")
        }
        try {
            val code = connection.responseCode
            check(code == 200 || code == 206) { "Download failed (HTTP $code)" }
            val resume = code == 206                              // otherwise the server starts again from 0
            connection.inputStream.use { input ->
                FileOutputStream(part, resume).use { out ->
                    val buffer = ByteArray(1 shl 16)
                    var done = if (resume) have else 0L
                    var shown = -1
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        out.write(buffer, 0, read)
                        done += read
                        val percent = (100 * done / model.bytes).toInt()
                        if (percent != shown) {
                            shown = percent
                            onProgress(done.toFloat() / model.bytes)
                        }
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
    }
}
