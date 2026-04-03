plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.bro.signtalk"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.bro.signtalk"
        minSdk = 29

        // 2. targetSdk도 compileSdk와 똑같이 35로 맞춰라 이말이지 유남생?!?
        targetSdk = 35

        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    implementation("androidx.core:core-telecom:1.0.0-alpha03")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.0")
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)

    // [핵심] 이거 하나면 충분하다 이말이야! 중복된 com.github...은 지워버려!
    implementation("io.github.webrtc-sdk:android:125.6422.06.1")
    implementation("com.google.android.material:material:1.9.0")
// 최신 버전으로 기강 잡아라!
    implementation("com.github.yalantis:ucrop:2.2.8")

    // CameraX
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")
    // CameraX ListenableFuture 지원
    implementation("com.google.guava:guava:31.0.1-android")
    // OkHttp (WebSocket + HTTP)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}