// JNI bridge to whisper.cpp for io.github.luismpso.spectre.WhisperNative.
#include <jni.h>
#include <string>

#include "whisper.h"

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_luismpso_spectre_WhisperNative_init(JNIEnv *env, jobject, jstring jpath) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = false;
    whisper_context *ctx = whisper_init_from_file_with_params(path, params);
    env->ReleaseStringUTFChars(jpath, path);
    return reinterpret_cast<jlong>(ctx);
}

// 16 kHz mono samples → UTF-8 text (bytes, so that any character survives the trip to Kotlin)
extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_github_luismpso_spectre_WhisperNative_transcribe(JNIEnv *env, jobject, jlong handle, jfloatArray jaudio,
                                                         jstring jlanguage, jint threads, jint audio_ctx,
                                                         jint max_tokens) {
    auto *ctx = reinterpret_cast<whisper_context *>(handle);
    const char *lang = env->GetStringUTFChars(jlanguage, nullptr);
    std::string language(lang);
    env->ReleaseStringUTFChars(jlanguage, lang);

    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.language = language.c_str();       // "auto" detects it
    p.n_threads = threads;
    p.translate = false;
    p.no_context = true;                 // every turn on its own: no text carried over
    p.no_timestamps = true;
    p.single_segment = true;
    p.print_special = false;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.suppress_blank = true;
    p.suppress_nst = true;               // no "(música)", "[risos]"…
    p.audio_ctx = audio_ctx;             // encode only the turn, not 30 s (0 = 30 s)
    p.max_tokens = max_tokens;           // a loop ("Então vamos. Então vamos. …") stops here

    jsize n = env->GetArrayLength(jaudio);
    jfloat *audio = env->GetFloatArrayElements(jaudio, nullptr);
    int rc = whisper_full(ctx, p, audio, n);
    env->ReleaseFloatArrayElements(jaudio, audio, JNI_ABORT);

    std::string text;
    if (rc == 0) {
        for (int i = 0; i < whisper_full_n_segments(ctx); ++i) text += whisper_full_get_segment_text(ctx, i);
    }
    jbyteArray out = env->NewByteArray(static_cast<jsize>(text.size()));
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(text.size()), reinterpret_cast<const jbyte *>(text.data()));
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_luismpso_spectre_WhisperNative_free(JNIEnv *, jobject, jlong handle) {
    whisper_free(reinterpret_cast<whisper_context *>(handle));
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_luismpso_spectre_WhisperNative_systemInfo(JNIEnv *env, jobject) {
    return env->NewStringUTF(whisper_print_system_info());
}
