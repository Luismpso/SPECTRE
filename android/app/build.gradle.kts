import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "io.github.luismpso.spectre"
    compileSdk = 36
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "io.github.luismpso.spectre"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }
    // one APK per processor: phones (arm64-v8a) and the Android Studio emulator on a PC (x86_64)
    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("debug")    // a personal app: installable without a keystore
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { compose = true }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }
    // compressed native libraries: one APK of ~26 MB that can be sent in a chat (installed, they take ~40 MB more)
    packaging { jniLibs { useLegacyPackaging = true } }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

// the voice model comes from your trained checkpoint: python -m spectre.export_android (in the repository)
val checkVoiceModel by tasks.registering {
    val assets = layout.projectDirectory.dir("src/main/assets")
    doLast {
        for (name in listOf("spectre_ecapa.onnx", "voices.json")) check(assets.file(name).asFile.exists()) {
            "Missing app/src/main/assets/$name: run `python -m spectre.export_android` in the repository first"
        }
    }
}
tasks.named("preBuild") { dependsOn(checkVoiceModel) }

dependencies {
    implementation(project(":core"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.onnxruntime.android)
}
