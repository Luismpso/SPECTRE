package io.github.luismpso.spectre

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.github.luismpso.spectre.core.Embedder
import java.nio.FloatBuffer

/** The SPECTRE voice model (spectre_ecapa.onnx: 16 kHz waveform → unit 192-d embedding) on ONNX Runtime. */
class OnnxEmbedder(model: ByteArray) : Embedder, AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(model, OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(2)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    })

    override fun embed(wavs: List<FloatArray>): List<FloatArray> = wavs.map { wav ->
        OnnxTensor.createTensor(env, FloatBuffer.wrap(wav), longArrayOf(1, wav.size.toLong())).use { input ->
            session.run(mapOf("wav" to input)).use { out ->
                @Suppress("UNCHECKED_CAST")
                (out[0].value as Array<FloatArray>)[0]
            }
        }
    }

    override fun close() = session.close()
}
