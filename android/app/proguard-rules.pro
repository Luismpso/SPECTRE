# ONNX Runtime creates and reads its Java objects from native code
-keep class ai.onnxruntime.** { *; }
# whisper.cpp through JNI (src/main/cpp/whisper_jni.cpp)
-keep class io.github.luismpso.spectre.WhisperNative { *; }
