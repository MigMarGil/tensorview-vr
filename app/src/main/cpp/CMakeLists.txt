cmake_minimum_required(VERSION 3.22.1)
project(tensorview_vr LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Khronos OpenXR loader, delivered as an Android prefab package (see app/build.gradle)
find_package(OpenXR REQUIRED CONFIG)

# NDK native_app_glue (lifecycle glue for NativeActivity)
set(APP_GLUE_DIR ${ANDROID_NDK}/sources/android/native_app_glue)
add_library(native_app_glue STATIC ${APP_GLUE_DIR}/android_native_app_glue.c)
target_include_directories(native_app_glue PUBLIC ${APP_GLUE_DIR})

add_library(tensorview_vr SHARED
    xr_app.cpp
    gl_batch.cpp
    audio.cpp
    onnx_model.cpp)

target_compile_options(tensorview_vr PRIVATE -O2 -Wall)
target_link_libraries(tensorview_vr native_app_glue OpenXR::openxr_loader android log EGL GLESv3 aaudio)
# Make sure NativeActivity can find its entry point inside the static glue library
set_property(TARGET tensorview_vr APPEND_STRING PROPERTY LINK_FLAGS " -u ANativeActivity_onCreate")
