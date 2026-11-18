#!/usr/bin/env bash
# Builds the Quest APK. Needs: JDK 17, Gradle 8.4, Android SDK (ANDROID_HOME), NDK 25.2.9519653, CMake 3.22.1.
set -euo pipefail
cd "$(dirname "$0")"

[ -f local.properties ] || echo "sdk.dir=${ANDROID_HOME:?set ANDROID_HOME first}" > local.properties

if [ ! -f release.keystore ]; then
  echo "==> Generating signing key (release.keystore)"
  keytool -genkeypair -keystore release.keystore -alias tensorview -keyalg RSA -keysize 2048 -validity 10000 \
    -storepass tensorview -keypass tensorview -dname "CN=TensorView VR, O=TensorView, C=ES"
fi

if command -v sdkmanager >/dev/null 2>&1; then
  echo "==> Ensuring Android platform 34 / build-tools 34 (needed for targetSdk 34)"
  yes | sdkmanager --licenses >/dev/null 2>&1 || true
  sdkmanager "platforms;android-34" "build-tools;34.0.0" >/dev/null || echo "WARN: sdkmanager failed - install platforms;android-34 manually"
else
  echo "WARN: sdkmanager not in PATH - make sure platforms;android-34 and build-tools;34.0.0 are installed"
fi

M=app/src/main/assets/models
if [ ! -f $M/vision_resnet_pro.onnx ] || [ ! -f $M/mobilenet_lite.onnx ] || [ ! -f $M/tiny_transformer.onnx ]; then
  echo "==> Generating sample ONNX models"
  [ -d venv ] || python3 -m venv venv
  ./venv/bin/pip install -q onnx numpy
  ./venv/bin/python tools/make_model.py $M
fi

gradle assembleRelease --no-daemon
echo "==> APK: $(pwd)/app/build/outputs/apk/release/app-release.apk"
