cmake_minimum_required(VERSION 3.22.1)
project(spectre_whisper C CXX)

# Whisper is unusably slow without optimisation, so optimise even the debug builds of the app.
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
if (ANDROID_ABI STREQUAL "arm64-v8a")
    # dot-product and half-precision instructions: every phone since ~2019 (and all from 2022 on) has them
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif ()

include(FetchContent)
FetchContent_Declare(whisper
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG v1.9.5
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(whisper)

add_library(spectre_whisper SHARED whisper_jni.cpp)
target_compile_options(spectre_whisper PRIVATE -O3)
target_link_libraries(spectre_whisper PRIVATE whisper android log)
