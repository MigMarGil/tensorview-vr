// TensorView VR - native OpenXR (OpenGL ES 3) app for Meta Quest.
// Hands-first: XR_EXT_hand_tracking pinch + far-field ray, with Touch-controller fallback.
#include <android/asset_manager.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <jni.h>

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#define XR_USE_PLATFORM_ANDROID
#define XR_USE_GRAPHICS_API_OPENGL_ES
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "audio.h"
#include "gl_batch.h"
#include "onnx_model.h"
#include "tv_math.h"

#define LOG_TAG "TensorViewVR"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define XR_OK(call)                                                   \
    do {                                                              \
        XrResult r__ = (call);                                        \
        if (XR_FAILED(r__)) { LOGE("%s failed: %d", #call, (int)r__); return false; } \
    } while (0)

namespace {
using namespace tv;

const V3 kGraphPos{0.0f, 0.10f, -1.55f};  // default graph center in LOCAL space (meters)
const float kButtonY = -0.55f, kHudZ = -1.30f;
const char* kTutorial[5] = {
    "RAISE YOUR HANDS IN FRONT OF YOU",
    "POINT AT A BLOCK WITH YOUR HAND",
    "PINCH THUMB + INDEX TO SELECT IT",
    "PINCH EMPTY SPACE + DRAG TO MOVE. BOTH HANDS TO ZOOM",
    "TAP A MODEL BUTTON TO COMPARE ARCHITECTURES"};
const XrPosef kIdentityPose = {{0, 0, 0, 1}, {0, 0, 0}};

struct Eye {
    XrSwapchain swapchain = XR_NULL_HANDLE;
    int32_t w = 0, h = 0;
    std::vector<XrSwapchainImageOpenGLESKHR> images;
    GLuint depth = 0;
};

struct Pointer {
    bool valid = false, usingHands = false, pinch = false, prevPinch = false;
    V3 origin, dir;
    int hover = -1, hoverBtn = -1, prevHover = -1, prevHoverBtn = -1;
    float dist = 3.0f;
    V3 prevOrigin;
    bool hasPrev = false, grabbing = false;
    bool hasJoints = false;
    V3 joint[XR_HAND_JOINT_COUNT_EXT];
    bool jointOk[XR_HAND_JOINT_COUNT_EXT] = {};
};

std::string human(double v, const char* unit) {
    char b[48];
    if (v >= 1e9) std::snprintf(b, sizeof b, "%.2fG%s", v / 1e9, unit);
    else if (v >= 1e6) std::snprintf(b, sizeof b, "%.2fM%s", v / 1e6, unit);
    else if (v >= 1e3) std::snprintf(b, sizeof b, "%.1fK%s", v / 1e3, unit);
    else std::snprintf(b, sizeof b, "%.0f%s", v, unit);
    return b;
}

std::string fit(const std::string& s, size_t n) { return s.size() <= n ? s : s.substr(0, n - 1) + "~"; }

bool rayBox(V3 o, V3 d, V3 c, V3 h, float& tHit) {
    float tmin = 0.0f, tmax = 100.0f;
    const float oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z}, cc[3] = {c.x, c.y, c.z}, hh[3] = {h.x, h.y, h.z};
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(dd[i]) < 1e-8f) {
            if (oo[i] < cc[i] - hh[i] || oo[i] > cc[i] + hh[i]) return false;
        } else {
            float t1 = (cc[i] - hh[i] - oo[i]) / dd[i], t2 = (cc[i] + hh[i] - oo[i]) / dd[i];
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    tHit = tmin;
    return true;
}

struct App {
    android_app* app = nullptr;
    bool resumed = false, sessionRunning = false, exitRequested = false, ready = false;

    // EGL
    EGLDisplay eglDisplay = EGL_NO_DISPLAY;
    EGLConfig eglConfig = nullptr;
    EGLContext eglContext = EGL_NO_CONTEXT;
    EGLSurface eglSurface = EGL_NO_SURFACE;

    // OpenXR core
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId systemId = XR_NULL_SYSTEM_ID;
    XrSession session = XR_NULL_HANDLE;
    XrSpace localSpace = XR_NULL_HANDLE;
    XrSessionState state = XR_SESSION_STATE_UNKNOWN;
    Eye eyes[2];
    GLuint fbo = 0;
    bool srgbSwapchain = false;

    // Hand tracking (optional extension)
    bool handExt = false, handSupported = false;
    XrHandTrackerEXT hand[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
    PFN_xrCreateHandTrackerEXT pfnCreateHand = nullptr;
    PFN_xrDestroyHandTrackerEXT pfnDestroyHand = nullptr;
    PFN_xrLocateHandJointsEXT pfnLocateHand = nullptr;

    // Controller fallback
    XrActionSet actionSet = XR_NULL_HANDLE;
    XrAction aimAction = XR_NULL_HANDLE, selectAction = XR_NULL_HANDLE;
    XrPath handPath[2] = {XR_NULL_PATH, XR_NULL_PATH};
    XrSpace aimSpace[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
    bool actionsReady = false;

    // Runtime permission handling (hand tracking is a runtime permission on Horizon OS)
    JNIEnv* env = nullptr;
    bool permRequested = false, handsCreated = false;
    std::chrono::steady_clock::time_point permTime, lastHandTry;

    // App state
    GlRenderer gl;
    struct Source { std::string label, path, file; bool user; };
    std::vector<Source> sources;
    int modelIndex = 0;
    Model model;
    std::string modelError;
    V3 graphPos = kGraphPos;
    float graphScale = 1.0f, moved = 0.0f;
    bool switched = false;
    int tutorial = 0;
    Audio audio;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Pointer ptr[2];
    int selected = -1;
    float fps = 72.0f;
    std::chrono::steady_clock::time_point lastFrame = std::chrono::steady_clock::now();
    Batch solid, overlay;

    // ------------------------------------------------------------ init
    bool init() {
        if (!initInstance()) return false;
        if (!initEgl()) return false;
        if (!initSession()) return false;
        if (!initSwapchains()) return false;
        initActions();  // best effort; hand trackers are created lazily in ensureHandTrackers()
        if (!gl.init(srgbSwapchain)) { LOGE("GL renderer init failed"); return false; }
        glGenFramebuffers(1, &fbo);
        initSources();
        loadSource(0);
        if (!audio.init()) LOGI("Audio unavailable - continuing without sound");
        ready = true;
        LOGI("TensorView VR initialized: %zu nodes, hand tracking: %s", model.nodes.size(), handSupported ? "yes" : "no");
        return true;
    }

    bool initInstance() {
        PFN_xrInitializeLoaderKHR initLoader = nullptr;
        XR_OK(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", (PFN_xrVoidFunction*)&initLoader));
        XrLoaderInitInfoAndroidKHR li{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
        li.applicationVM = app->activity->vm;
        li.applicationContext = app->activity->clazz;
        XR_OK(initLoader((const XrLoaderInitInfoBaseHeaderKHR*)&li));

        uint32_t n = 0;
        XR_OK(xrEnumerateInstanceExtensionProperties(nullptr, 0, &n, nullptr));
        std::vector<XrExtensionProperties> props(n, XrExtensionProperties{XR_TYPE_EXTENSION_PROPERTIES});
        XR_OK(xrEnumerateInstanceExtensionProperties(nullptr, n, &n, props.data()));
        auto has = [&](const char* name) {
            for (auto& p : props) if (!std::strcmp(p.extensionName, name)) return true;
            return false;
        };
        if (!has(XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME) || !has(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME)) {
            LOGE("Required OpenXR extensions (OpenGL ES / Android) are missing");
            return false;
        }
        std::vector<const char*> exts = {XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME, XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME};
        handExt = has(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
        if (handExt) exts.push_back(XR_EXT_HAND_TRACKING_EXTENSION_NAME);

        XrInstanceCreateInfoAndroidKHR ia{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
        ia.applicationVM = app->activity->vm;
        ia.applicationActivity = app->activity->clazz;
        XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
        ci.next = &ia;
        std::strncpy(ci.applicationInfo.applicationName, "TensorView VR", XR_MAX_APPLICATION_NAME_SIZE - 1);
        std::strncpy(ci.applicationInfo.engineName, "TensorViewNative", XR_MAX_ENGINE_NAME_SIZE - 1);
        ci.applicationInfo.applicationVersion = 1;
        ci.applicationInfo.engineVersion = 1;
        ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 34);
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.enabledExtensionNames = exts.data();
        XR_OK(xrCreateInstance(&ci, &instance));

        XrSystemGetInfo si{XR_TYPE_SYSTEM_GET_INFO};
        si.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        XR_OK(xrGetSystem(instance, &si, &systemId));

        if (handExt) {
            XrSystemHandTrackingPropertiesEXT hp{XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};
            XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
            sp.next = &hp;
            if (XR_SUCCEEDED(xrGetSystemProperties(instance, systemId, &sp))) handSupported = hp.supportsHandTracking == XR_TRUE;
        }
        return true;
    }

    bool initEgl() {
        eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL_NO_DISPLAY || !eglInitialize(eglDisplay, nullptr, nullptr)) { LOGE("eglInitialize failed"); return false; }
        const EGLint cfg[] = {EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 0,
                              EGL_SURFACE_TYPE, EGL_WINDOW_BIT | EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
        EGLint n = 0;
        if (!eglChooseConfig(eglDisplay, cfg, &eglConfig, 1, &n) || n < 1) { LOGE("eglChooseConfig failed"); return false; }
        const EGLint ctx[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        eglContext = eglCreateContext(eglDisplay, eglConfig, EGL_NO_CONTEXT, ctx);
        if (eglContext == EGL_NO_CONTEXT) { LOGE("eglCreateContext failed"); return false; }
        const EGLint pb[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
        eglSurface = eglCreatePbufferSurface(eglDisplay, eglConfig, pb);
        if (eglSurface == EGL_NO_SURFACE) { LOGE("eglCreatePbufferSurface failed"); return false; }
        if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) { LOGE("eglMakeCurrent failed"); return false; }
        return true;
    }

    bool initSession() {
        PFN_xrGetOpenGLESGraphicsRequirementsKHR getReq = nullptr;
        XR_OK(xrGetInstanceProcAddr(instance, "xrGetOpenGLESGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&getReq));
        XrGraphicsRequirementsOpenGLESKHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
        XR_OK(getReq(instance, systemId, &req));

        XrGraphicsBindingOpenGLESAndroidKHR gb{XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
        gb.display = eglDisplay;
        gb.config = eglConfig;
        gb.context = eglContext;
        XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
        sci.next = &gb;
        sci.systemId = systemId;
        XR_OK(xrCreateSession(instance, &sci, &session));

        XrReferenceSpaceCreateInfo rsi{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        rsi.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
        rsi.poseInReferenceSpace = kIdentityPose;
        XR_OK(xrCreateReferenceSpace(session, &rsi, &localSpace));
        return true;
    }

    bool initSwapchains() {
        uint32_t n = 0;
        XR_OK(xrEnumerateViewConfigurationViews(instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &n, nullptr));
        if (n != 2) { LOGE("Expected 2 views, got %u", n); return false; }
        XrViewConfigurationView vcv[2] = {{XR_TYPE_VIEW_CONFIGURATION_VIEW}, {XR_TYPE_VIEW_CONFIGURATION_VIEW}};
        XR_OK(xrEnumerateViewConfigurationViews(instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 2, &n, vcv));

        uint32_t fc = 0;
        XR_OK(xrEnumerateSwapchainFormats(session, 0, &fc, nullptr));
        std::vector<int64_t> formats(fc);
        XR_OK(xrEnumerateSwapchainFormats(session, fc, &fc, formats.data()));
        int64_t chosen = formats.empty() ? GL_RGBA8 : formats[0];
        bool found = false;
        for (int64_t f : formats) if (f == GL_SRGB8_ALPHA8) { chosen = f; found = true; break; }
        if (!found) for (int64_t f : formats) if (f == GL_RGBA8) { chosen = f; break; }
        srgbSwapchain = (chosen == GL_SRGB8_ALPHA8);

        for (int i = 0; i < 2; ++i) {
            Eye& e = eyes[i];
            e.w = (int32_t)vcv[i].recommendedImageRectWidth;
            e.h = (int32_t)vcv[i].recommendedImageRectHeight;
            XrSwapchainCreateInfo sc{XR_TYPE_SWAPCHAIN_CREATE_INFO};
            sc.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
            sc.format = chosen;
            sc.sampleCount = 1;
            sc.width = (uint32_t)e.w;
            sc.height = (uint32_t)e.h;
            sc.faceCount = 1;
            sc.arraySize = 1;
            sc.mipCount = 1;
            XR_OK(xrCreateSwapchain(session, &sc, &e.swapchain));
            uint32_t cnt = 0;
            XR_OK(xrEnumerateSwapchainImages(e.swapchain, 0, &cnt, nullptr));
            e.images.assign(cnt, XrSwapchainImageOpenGLESKHR{XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR});
            XR_OK(xrEnumerateSwapchainImages(e.swapchain, cnt, &cnt, (XrSwapchainImageBaseHeader*)e.images.data()));
            glGenRenderbuffers(1, &e.depth);
            glBindRenderbuffer(GL_RENDERBUFFER, e.depth);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, e.w, e.h);
        }
        LOGI("Swapchains %dx%d, format %s", eyes[0].w, eyes[0].h, srgbSwapchain ? "SRGB8_ALPHA8" : "RGBA8");
        return true;
    }

    void initActions() {
        XrActionSetCreateInfo asi{XR_TYPE_ACTION_SET_CREATE_INFO};
        std::strcpy(asi.actionSetName, "tensorview");
        std::strcpy(asi.localizedActionSetName, "TensorView");
        if (XR_FAILED(xrCreateActionSet(instance, &asi, &actionSet))) return;
        xrStringToPath(instance, "/user/hand/left", &handPath[0]);
        xrStringToPath(instance, "/user/hand/right", &handPath[1]);

        XrActionCreateInfo ai{XR_TYPE_ACTION_CREATE_INFO};
        ai.countSubactionPaths = 2;
        ai.subactionPaths = handPath;
        ai.actionType = XR_ACTION_TYPE_POSE_INPUT;
        std::strcpy(ai.actionName, "aim_pose");
        std::strcpy(ai.localizedActionName, "Aim pose");
        if (XR_FAILED(xrCreateAction(actionSet, &ai, &aimAction))) return;
        ai.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
        std::strcpy(ai.actionName, "select");
        std::strcpy(ai.localizedActionName, "Select");
        if (XR_FAILED(xrCreateAction(actionSet, &ai, &selectAction))) return;

        auto path = [&](const char* s) { XrPath p = XR_NULL_PATH; xrStringToPath(instance, s, &p); return p; };
        {
            XrActionSuggestedBinding b[4] = {{aimAction, path("/user/hand/left/input/aim/pose")},
                                             {aimAction, path("/user/hand/right/input/aim/pose")},
                                             {selectAction, path("/user/hand/left/input/trigger/value")},
                                             {selectAction, path("/user/hand/right/input/trigger/value")}};
            XrInteractionProfileSuggestedBinding sb{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
            sb.interactionProfile = path("/interaction_profiles/oculus/touch_controller");
            sb.countSuggestedBindings = 4;
            sb.suggestedBindings = b;
            xrSuggestInteractionProfileBindings(instance, &sb);
        }
        XrSessionActionSetsAttachInfo att{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
        att.countActionSets = 1;
        att.actionSets = &actionSet;
        if (XR_FAILED(xrAttachSessionActionSets(session, &att))) return;
        for (int h = 0; h < 2; ++h) {
            XrActionSpaceCreateInfo si{XR_TYPE_ACTION_SPACE_CREATE_INFO};
            si.action = aimAction;
            si.subactionPath = handPath[h];
            si.poseInActionSpace = kIdentityPose;
            if (XR_FAILED(xrCreateActionSpace(session, &si, &aimSpace[h]))) return;
        }
        actionsReady = true;
    }

    bool hasPermission(const char* perm) {
        if (!env) return false;
        jobject act = app->activity->clazz;
        jclass cls = env->GetObjectClass(act);
        jmethodID m = env->GetMethodID(cls, "checkSelfPermission", "(Ljava/lang/String;)I");
        bool ok = false;
        if (m) {
            jstring js = env->NewStringUTF(perm);
            ok = env->CallIntMethod(act, m, js) == 0;  // PackageManager.PERMISSION_GRANTED
            env->DeleteLocalRef(js);
        }
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(cls);
        return ok;
    }

    void requestPermissions(const char* const* perms, int n) {
        if (!env) return;
        jobject act = app->activity->clazz;
        jclass cls = env->GetObjectClass(act);
        jmethodID m = env->GetMethodID(cls, "requestPermissions", "([Ljava/lang/String;I)V");
        jclass strCls = env->FindClass("java/lang/String");
        if (m && strCls) {
            jobjectArray arr = env->NewObjectArray(n, strCls, nullptr);
            for (int i = 0; i < n; ++i) {
                jstring js = env->NewStringUTF(perms[i]);
                env->SetObjectArrayElement(arr, i, js);
                env->DeleteLocalRef(js);
            }
            env->CallVoidMethod(act, m, arr, 1);
            env->DeleteLocalRef(arr);
        }
        if (env->ExceptionCheck()) env->ExceptionClear();
        if (strCls) env->DeleteLocalRef(strCls);
        env->DeleteLocalRef(cls);
    }

    // Creates the XR_EXT_hand_tracking trackers once the permission is granted (retries every 2 s).
    void ensureHandTrackers() {
        if (handsCreated || !handExt) return;
        const auto now = std::chrono::steady_clock::now();
        const bool granted = hasPermission("com.oculus.permission.HAND_TRACKING") || hasPermission("horizonos.permission.HAND_TRACKING");
        if (!granted && !permRequested) {
            permRequested = true;
            permTime = now;
            const char* perms[2] = {"com.oculus.permission.HAND_TRACKING", "horizonos.permission.HAND_TRACKING"};
            requestPermissions(perms, 2);
            LOGI("Requested hand tracking permission");
            return;
        }
        // If the permission dialog never resolves (or the OS does not gate it), try anyway after a short wait.
        if (!granted && std::chrono::duration<float>(now - permTime).count() < 2.0f) return;
        if (std::chrono::duration<float>(now - lastHandTry).count() < 2.0f) return;
        lastHandTry = now;
        initHandTracking();
        handsCreated = hand[0] != XR_NULL_HANDLE || hand[1] != XR_NULL_HANDLE;
        if (handsCreated) LOGI("Hand trackers created");
    }

    void initHandTracking() {
        if (!handExt) return;
        if (!pfnCreateHand) xrGetInstanceProcAddr(instance, "xrCreateHandTrackerEXT", (PFN_xrVoidFunction*)&pfnCreateHand);
        if (!pfnDestroyHand) xrGetInstanceProcAddr(instance, "xrDestroyHandTrackerEXT", (PFN_xrVoidFunction*)&pfnDestroyHand);
        if (!pfnLocateHand) xrGetInstanceProcAddr(instance, "xrLocateHandJointsEXT", (PFN_xrVoidFunction*)&pfnLocateHand);
        if (!pfnCreateHand || !pfnLocateHand) return;
        for (int h = 0; h < 2; ++h) {
            if (hand[h] != XR_NULL_HANDLE) continue;
            XrHandTrackerCreateInfoEXT ci{XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT};
            ci.hand = h == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
            ci.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
            if (XR_FAILED(pfnCreateHand(session, &ci, &hand[h]))) hand[h] = XR_NULL_HANDLE;
        }
    }

    void initSources() {
        sources = {{"RESNET CNN", "models/vision_resnet_pro.onnx", "vision_resnet_pro.onnx", false},
                   {"MOBILENET", "models/mobilenet_lite.onnx", "mobilenet_lite.onnx", false},
                   {"TRANSFORMER", "models/tiny_transformer.onnx", "tiny_transformer.onnx", false}};
        // Optional user model: adb push my.onnx /sdcard/Android/data/com.migmargv.tensorviewvr/files/model.onnx
        if (app->activity->externalDataPath) {
            const std::string p = std::string(app->activity->externalDataPath) + "/model.onnx";
            if (std::ifstream(p, std::ios::binary)) sources.push_back({"MY MODEL", p, "model.onnx (user)", true});
        }
    }

    void loadSource(int i) {
        modelIndex = i;
        selected = -1;
        const Source& src = sources[i];
        std::vector<uint8_t> bytes;
        if (src.user) {
            std::ifstream f(src.path, std::ios::binary);
            if (f) bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        } else {
            AAsset* a = AAssetManager_open(app->activity->assetManager, src.path.c_str(), AASSET_MODE_BUFFER);
            if (a) {
                bytes.resize((size_t)AAsset_getLength(a));
                AAsset_read(a, bytes.data(), bytes.size());
                AAsset_close(a);
            }
        }
        model = Model();
        modelError.clear();
        if (bytes.empty()) { modelError = "model file not found"; return; }
        if (!parseOnnx(bytes.data(), bytes.size(), src.file, model, modelError)) {
            LOGE("ONNX parse error: %s", modelError.c_str());
            model = Model();
        }
    }

    // ------------------------------------------------------------ events
    void pollEvents() {
        XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(instance, &ev) == XR_SUCCESS) {
            if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                auto* e = (XrEventDataSessionStateChanged*)&ev;
                state = e->state;
                LOGI("Session state -> %d", (int)state);
                if (state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
                    bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    sessionRunning = XR_SUCCEEDED(xrBeginSession(session, &bi));
                } else if (state == XR_SESSION_STATE_STOPPING) {
                    xrEndSession(session);
                    sessionRunning = false;
                } else if (state == XR_SESSION_STATE_EXITING || state == XR_SESSION_STATE_LOSS_PENDING) {
                    exitRequested = true;
                }
            } else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
                exitRequested = true;
            }
            ev = XrEventDataBuffer{XR_TYPE_EVENT_DATA_BUFFER};
        }
    }

    // ------------------------------------------------------------ input
    bool updateHand(int h, XrTime t, const V3& head, const Quat& headQ) {
        Pointer& p = ptr[h];
        if (hand[h] == XR_NULL_HANDLE || !pfnLocateHand) return false;
        XrHandJointsLocateInfoEXT li{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};
        li.baseSpace = localSpace;
        li.time = t;
        XrHandJointLocationEXT jl[XR_HAND_JOINT_COUNT_EXT];
        XrHandJointLocationsEXT locs{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};
        locs.jointCount = XR_HAND_JOINT_COUNT_EXT;
        locs.jointLocations = jl;
        if (XR_FAILED(pfnLocateHand(hand[h], &li, &locs)) || !locs.isActive) return false;
        for (uint32_t j = 0; j < XR_HAND_JOINT_COUNT_EXT; ++j) {
            p.jointOk[j] = (jl[j].locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;
            p.joint[j] = {jl[j].pose.position.x, jl[j].pose.position.y, jl[j].pose.position.z};
        }
        if (!p.jointOk[XR_HAND_JOINT_THUMB_TIP_EXT] || !p.jointOk[XR_HAND_JOINT_INDEX_TIP_EXT]) return false;
        const V3 th = p.joint[XR_HAND_JOINT_THUMB_TIP_EXT], ix = p.joint[XR_HAND_JOINT_INDEX_TIP_EXT];
        const float d = len(th - ix);
        p.pinch = p.prevPinch ? d < 0.032f : d < 0.022f;  // hysteresis
        const V3 pt = (th + ix) * 0.5f;
        // far-field ray: from an estimated shoulder through the pinch point
        V3 fwd = rotate(headQ, V3{0, 0, -1});
        fwd.y = 0;
        fwd = norm(fwd);
        const V3 right = cross(fwd, V3{0, 1, 0});
        const V3 shoulder = head + right * (h == 1 ? 0.17f : -0.17f) + V3{0, -0.20f, 0};
        p.origin = pt;
        p.dir = norm(pt - shoulder);
        p.valid = p.hasJoints = p.usingHands = true;
        return true;
    }

    bool updateController(int h, XrTime t) {
        Pointer& p = ptr[h];
        if (!actionsReady || state != XR_SESSION_STATE_FOCUSED) return false;
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.subactionPath = handPath[h];
        gi.action = aimAction;
        XrActionStatePose ps{XR_TYPE_ACTION_STATE_POSE};
        if (XR_FAILED(xrGetActionStatePose(session, &gi, &ps)) || !ps.isActive) return false;
        XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
        if (XR_FAILED(xrLocateSpace(aimSpace[h], localSpace, t, &loc))) return false;
        const XrSpaceLocationFlags need = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
        if ((loc.locationFlags & need) != need) return false;
        const Quat q{loc.pose.orientation.x, loc.pose.orientation.y, loc.pose.orientation.z, loc.pose.orientation.w};
        p.origin = {loc.pose.position.x, loc.pose.position.y, loc.pose.position.z};
        p.dir = norm(rotate(q, V3{0, 0, -1}));
        gi.action = selectAction;
        XrActionStateFloat fs{XR_TYPE_ACTION_STATE_FLOAT};
        p.pinch = XR_SUCCEEDED(xrGetActionStateFloat(session, &gi, &fs)) && fs.isActive && fs.currentState > (p.prevPinch ? 0.4f : 0.6f);
        p.valid = true;
        p.usingHands = false;
        return true;
    }

    // ------------------------------------------------------------ interaction helpers
    V3 nodeCenter(const GraphNode& n) const { return graphPos + V3{n.x, n.y, n.z} * graphScale; }
    V3 nodeHalf(const GraphNode& n) const { return V3{n.hx, n.hy, n.hz} * graphScale; }
    float timeSec() const { return std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count(); }

    int buttonCount() const { return (int)sources.size() + 1; }  // models + RESET VIEW
    V3 buttonCenter(int i) const {
        const int n = buttonCount();
        const float w = 0.30f, gap = 0.04f, total = n * w + (n - 1) * gap;
        return {-total * 0.5f + w * 0.5f + i * (w + gap), kButtonY, kHudZ};
    }
    std::string buttonLabel(int i) const { return i < (int)sources.size() ? sources[i].label : "RESET VIEW"; }

    void pressButton(int i) {
        if (i < (int)sources.size()) {
            if (i != modelIndex) {
                loadSource(i);
                switched = true;
                for (auto& q : ptr) { q.hover = -1; q.prevHover = -1; }  // node ids belong to the old model
            }
        } else {
            graphPos = kGraphPos;
            graphScale = 1.0f;
        }
        audio.play(Sfx::Switch);
    }

    void advanceTutorial() {
        if (tutorial >= 5) return;
        const bool anyValid = ptr[0].valid || ptr[1].valid;
        const bool anyHover = ptr[0].hover >= 0 || ptr[1].hover >= 0;
        bool ok = false;
        switch (tutorial) {
            case 0: ok = anyValid; break;
            case 1: ok = anyHover; break;
            case 2: ok = selected >= 0; break;
            case 3: ok = moved > 0.25f || std::fabs(std::log(graphScale)) > 0.15f; break;
            case 4: ok = switched; break;
        }
        if (ok) { ++tutorial; audio.play(Sfx::Step); }
    }

    void updateInput(XrTime t, const V3& head, const Quat& headQ) {
        if (actionsReady && state == XR_SESSION_STATE_FOCUSED) {
            XrActiveActionSet aas{actionSet, XR_NULL_PATH};
            XrActionsSyncInfo si{XR_TYPE_ACTIONS_SYNC_INFO};
            si.countActiveActionSets = 1;
            si.activeActionSets = &aas;
            xrSyncActions(session, &si);
        }
        // 1) poses + hover (nodes and buttons, nearest wins)
        for (int h = 0; h < 2; ++h) {
            Pointer& p = ptr[h];
            p.valid = p.hasJoints = false;
            if (!updateHand(h, t, head, headQ)) updateController(h, t);
            p.hover = p.hoverBtn = -1;
            p.dist = 3.0f;
            if (!p.valid) { p.pinch = false; continue; }
            for (const auto& n : model.nodes) {
                float th;
                if (rayBox(p.origin, p.dir, nodeCenter(n), nodeHalf(n), th) && th < p.dist) { p.dist = th; p.hover = n.id; }
            }
            for (int i = 0; i < buttonCount(); ++i) {
                float th;
                if (rayBox(p.origin, p.dir, buttonCenter(i), V3{0.15f, 0.035f, 0.015f}, th) && th < p.dist) {
                    p.dist = th; p.hover = -1; p.hoverBtn = i;
                }
            }
        }
        // 2) discrete actions + sounds
        for (int h = 0; h < 2; ++h) {
            Pointer& p = ptr[h];
            if (p.valid && p.pinch && !p.prevPinch) {
                if (p.hoverBtn >= 0) pressButton(p.hoverBtn);
                else if (p.hover >= 0) { selected = p.hover; audio.play(Sfx::Select); }
                else { p.grabbing = true; audio.play(Sfx::Grab); }
            }
            if ((!p.valid || !p.pinch) && p.grabbing) { p.grabbing = false; audio.play(Sfx::Release); }
            if (p.valid && (p.hover != p.prevHover || p.hoverBtn != p.prevHoverBtn) && (p.hover >= 0 || p.hoverBtn >= 0))
                audio.play(Sfx::Hover);
        }
        // 3) grab gestures: one hand moves the graph, two hands move + zoom it
        Pointer* g[2] = {nullptr, nullptr};
        int ng = 0;
        for (int h = 0; h < 2; ++h) if (ptr[h].grabbing && ptr[h].valid && ptr[h].hasPrev) g[ng++] = &ptr[h];
        if (ng == 1) {
            const V3 d = (g[0]->origin - g[0]->prevOrigin) * 2.5f;
            graphPos = graphPos + d;
            moved += len(d);
        } else if (ng == 2) {
            const float d0 = len(g[0]->prevOrigin - g[1]->prevOrigin), d1 = len(g[0]->origin - g[1]->origin);
            if (d0 > 0.03f) graphScale *= std::min(1.1f, std::max(0.9f, d1 / d0));
            const V3 d = ((g[0]->origin - g[0]->prevOrigin) + (g[1]->origin - g[1]->prevOrigin)) * 1.25f;
            graphPos = graphPos + d;
            moved += len(d);
        }
        graphScale = std::min(2.5f, std::max(0.35f, graphScale));
        graphPos.x = std::min(1.4f, std::max(-1.4f, graphPos.x));
        graphPos.y = std::min(1.0f, std::max(-0.6f, graphPos.y));
        graphPos.z = std::min(-0.7f, std::max(-3.2f, graphPos.z));
        for (int h = 0; h < 2; ++h) {
            Pointer& p = ptr[h];
            p.prevOrigin = p.origin;
            p.hasPrev = p.valid;
            p.prevPinch = p.pinch;
            p.prevHover = p.hover;
            p.prevHoverBtn = p.hoverBtn;
        }
        advanceTutorial();
    }

    // ------------------------------------------------------------ scene
    void buildScene() {
        solid.clear();
        overlay.clear();
        const int nNodes = (int)model.nodes.size();
        int hov = ptr[0].hover >= 0 ? ptr[0].hover : ptr[1].hover;
        if (hov >= nNodes) hov = -1;
        if (selected >= nNodes) selected = -1;
        const float time = timeSec(), gs = graphScale;
        const Color gold{1.0f, 0.88f, 0.3f, 1};

        // edges (+ animated data-flow pulses)
        for (const auto& n : model.nodes)
            for (int c : n.outputs) {
                const auto& m = model.nodes[c];
                const bool hot = (n.id == selected || m.id == selected);
                const V3 cn = nodeCenter(n), cm = nodeCenter(m), d = cm - cn;
                V3 a, b;
                if (std::fabs(d.x) >= std::fabs(d.y)) {  // horizontal flow (either direction)
                    const float sx = d.x >= 0 ? 1.0f : -1.0f;
                    a = cn + V3{sx * n.hx * gs, 0, 0};
                    b = cm - V3{sx * m.hx * gs, 0, 0};
                } else {                                 // vertical hop between snake rows
                    const float sy = d.y >= 0 ? 1.0f : -1.0f;
                    a = cn + V3{0, sy * n.hy * gs, 0};
                    b = cm - V3{0, sy * m.hy * gs, 0};
                }
                solid.beam(a, b, (hot ? 0.008f : 0.004f) * gs, hot ? gold : Color{0.40f, 0.52f, 0.70f, 1});
                const float ph = std::fmod(time * 0.30f + 0.137f * (float)(n.id * 5 + c), 1.0f);
                const float ps = 0.007f * gs;
                solid.box(a + (b - a) * ph, {ps, ps, ps}, hot ? Color{1, 1, 0.7f, 1} : Color{0.55f, 0.85f, 1.0f, 1});
            }
        // nodes
        for (const auto& n : model.nodes) {
            const V3 c = nodeCenter(n), h = nodeHalf(n);
            const bool isSel = n.id == selected, isHov = n.id == hov;
            const float k = isHov ? 1.35f : 1.0f;
            solid.box(c, h, {std::min(1.0f, n.r * k), std::min(1.0f, n.g * k), std::min(1.0f, n.b * k), 1});
            if (isSel) solid.boxFrame(c, h + V3{0.008f, 0.008f, 0.008f} * gs, 0.006f * gs, {1.0f, 0.9f, 0.2f, 1});
            else if (isHov) solid.boxFrame(c, h + V3{0.006f, 0.006f, 0.006f} * gs, 0.004f * gs, {1, 1, 1, 1});
        }
        // model buttons + reset
        for (int i = 0; i < buttonCount(); ++i) {
            const bool hotb = ptr[0].hoverBtn == i || ptr[1].hoverBtn == i, active = i == modelIndex;
            const Color c = hotb ? Color{0.35f, 0.65f, 1.0f, 1} : (active ? Color{0.18f, 0.42f, 0.75f, 1} : Color{0.10f, 0.16f, 0.28f, 1});
            const V3 bc = buttonCenter(i);
            solid.box(bc, {0.15f, 0.035f, 0.01f}, c);
            const std::string lb = buttonLabel(i);
            const float tw = (float)lb.size() * 6 * 0.0034f;
            overlay.text(lb, {bc.x - tw * 0.5f, bc.y + 4 * 0.0034f, bc.z + 0.012f}, 0.0034f, {1, 1, 1, 1});
        }
        // hands / controllers / rays
        for (int h = 0; h < 2; ++h) {
            const Pointer& p = ptr[h];
            if (!p.valid) continue;
            if (p.hasJoints) {
                for (uint32_t j = 0; j < XR_HAND_JOINT_COUNT_EXT; ++j)
                    if (p.jointOk[j]) {
                        const float s = (j == XR_HAND_JOINT_THUMB_TIP_EXT || j == XR_HAND_JOINT_INDEX_TIP_EXT) ? 0.007f : 0.0045f;
                        solid.box(p.joint[j], {s, s, s}, p.pinch ? Color{1.0f, 0.6f, 0.2f, 1} : Color{0.65f, 0.85f, 1.0f, 1});
                    }
            } else {
                solid.box(p.origin, {0.015f, 0.015f, 0.03f}, {0.7f, 0.8f, 0.9f, 1});
            }
            const bool hit = p.hover >= 0 || p.hoverBtn >= 0;
            const float d = hit ? p.dist : 2.5f;
            const Color rc = p.pinch ? Color{1.0f, 0.6f, 0.2f, 1} : Color{0.3f, 0.9f, 1.0f, 1};
            solid.beam(p.origin, p.origin + p.dir * d, 0.0025f, rc);
            const float rs = hit ? 0.012f : 0.008f;
            solid.box(p.origin + p.dir * d, {rs, rs, rs}, rc);
        }

        // HUD (overlay, drawn after solids, no depth write)
        const float px = 0.0036f;
        const V3 pc{0.0f, -0.88f, kHudZ};
        overlay.panel(pc, 1.46f, 0.38f, {0.02f, 0.03f, 0.06f, 0.85f});
        auto line = [&](int row, const std::string& s, Color c) {
            overlay.text(fit(s, 60), {pc.x - 0.70f, pc.y + 0.165f - row * 9.2f * px, pc.z + 0.002f}, px, c);
        };
        const Color white{0.92f, 0.95f, 1, 1}, dim{0.55f, 0.65f, 0.80f, 1};
        overlay.text("TENSORVIEW VR  -  SPATIAL NEURAL NETWORK PROFILER", {-0.70f, kGraphPos.y + 0.78f, -1.40f}, 0.0045f, {0.5f, 0.85f, 1.0f, 1});

        // first-five-minutes tutorial banner + progress bar
        const float frac = tutorial / 5.0f;
        overlay.panel({0.0f, -0.668f, kHudZ}, 1.4f, 0.010f, {0.10f, 0.14f, 0.22f, 0.9f});
        if (tutorial > 0) overlay.panel({-0.7f + 0.7f * frac * 2.0f * 0.5f, -0.668f, kHudZ + 0.001f}, 1.4f * frac, 0.010f, gold);
        if (tutorial < 5) overlay.text("STEP " + std::to_string(tutorial + 1) + "/5  " + kTutorial[tutorial], {-0.70f, -0.612f, kHudZ}, 0.0040f, gold);
        else overlay.text("READY - EXPLORE FREELY", {-0.70f, -0.612f, kHudZ}, 0.0040f, {0.4f, 1.0f, 0.6f, 1});

        if (model.nodes.empty()) {
            line(0, "MODEL LOAD FAILED: " + modelError, {1.0f, 0.4f, 0.4f, 1});
            line(2, "Push a valid .onnx to <app files>/model.onnx", dim);
        } else {
            line(0, "MODEL " + sources[modelIndex].file + "   NODES " + std::to_string(model.nodes.size()), white);
            line(1, "PARAMS " + human((double)model.total_params, "") + "   FLOPs " + human((double)model.total_flops, "") +
                        "   MEM " + human((double)model.total_mem_kb * 1024.0, "B") + "   FPS " + std::to_string((int)(fps + 0.5f)), dim);
            line(2, "------------------------------------------------------------", dim);
            if (selected >= 0) {
                const auto& n = model.nodes[selected];
                line(3, "SELECTED: " + n.name + "  (" + n.op + ")", gold);
                line(4, "IN  " + n.in_shape, white);
                line(5, "OUT " + n.out_shape, white);
                line(6, "PARAMS " + human((double)n.params, "") + "   FLOPs " + human((double)n.flops, "") + "   MEM " + human((double)n.mem_kb * 1024.0, "B"), white);
            } else {
                line(3, "SELECT A LAYER TO SEE ITS DETAILS", gold);
            }
            if (hov >= 0 && hov != selected) line(7, "HOVER: " + model.nodes[hov].name + " (" + model.nodes[hov].op + ")", dim);
        }
    }

    // ------------------------------------------------------------ frame
    bool renderFrame(XrTime t, XrCompositionLayerProjection& layer, XrCompositionLayerProjectionView* pv) {
        XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
        XrViewState vs{XR_TYPE_VIEW_STATE};
        XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
        li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        li.displayTime = t;
        li.space = localSpace;
        uint32_t cnt = 0;
        if (XR_FAILED(xrLocateViews(session, &li, &vs, 2, &cnt, views)) || cnt != 2) return false;
        const XrViewStateFlags need = XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
        if ((vs.viewStateFlags & need) != need) return false;

        const V3 p0{views[0].pose.position.x, views[0].pose.position.y, views[0].pose.position.z};
        const V3 p1{views[1].pose.position.x, views[1].pose.position.y, views[1].pose.position.z};
        const Quat headQ{views[0].pose.orientation.x, views[0].pose.orientation.y, views[0].pose.orientation.z, views[0].pose.orientation.w};
        updateInput(t, (p0 + p1) * 0.5f, headQ);
        buildScene();

        for (int e = 0; e < 2; ++e) {
            Eye& eye = eyes[e];
            uint32_t idx = 0;
            XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            if (XR_FAILED(xrAcquireSwapchainImage(eye.swapchain, &ai, &idx))) return false;
            XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            wi.timeout = XR_INFINITE_DURATION;
            if (XR_FAILED(xrWaitSwapchainImage(eye.swapchain, &wi))) return false;

            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, eye.images[idx].image, 0);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, eye.depth);
            glViewport(0, 0, eye.w, eye.h);
            glDepthMask(GL_TRUE);
            glClearColor(0.004f, 0.007f, 0.016f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            const XrFovf& f = views[e].fov;
            const Mat4 proj = projectionFromFov(f.angleLeft, f.angleRight, f.angleUp, f.angleDown, 0.05f, 50.0f);
            const Quat q{views[e].pose.orientation.x, views[e].pose.orientation.y, views[e].pose.orientation.z, views[e].pose.orientation.w};
            const V3 pos{views[e].pose.position.x, views[e].pose.position.y, views[e].pose.position.z};
            const Mat4 vp = mul(proj, viewFromPose(pos, q));
            gl.draw(solid, vp, true);
            gl.draw(overlay, vp, false);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            xrReleaseSwapchainImage(eye.swapchain, &ri);

            pv[e] = XrCompositionLayerProjectionView{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            pv[e].pose = views[e].pose;
            pv[e].fov = views[e].fov;
            pv[e].subImage.swapchain = eye.swapchain;
            pv[e].subImage.imageRect = {{0, 0}, {eye.w, eye.h}};
            pv[e].subImage.imageArrayIndex = 0;
        }
        layer = XrCompositionLayerProjection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        layer.space = localSpace;
        layer.viewCount = 2;
        layer.views = pv;
        return true;
    }

    void frame() {
        XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState fs{XR_TYPE_FRAME_STATE};
        if (XR_FAILED(xrWaitFrame(session, &wi, &fs))) return;
        XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};
        if (XR_FAILED(xrBeginFrame(session, &bi))) return;

        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - lastFrame).count();
        lastFrame = now;
        if (dt > 0.0001f && dt < 1.0f) fps = fps * 0.95f + (1.0f / dt) * 0.05f;

        XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        XrCompositionLayerProjectionView pv[2];
        const XrCompositionLayerBaseHeader* layers[1] = {nullptr};
        uint32_t layerCount = 0;
        if (fs.shouldRender && renderFrame(fs.predictedDisplayTime, layer, pv)) {
            layers[0] = (const XrCompositionLayerBaseHeader*)&layer;
            layerCount = 1;
        }
        XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
        ei.displayTime = fs.predictedDisplayTime;
        ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        ei.layerCount = layerCount;
        ei.layers = layers;
        xrEndFrame(session, &ei);
    }

    // ------------------------------------------------------------ shutdown
    void shutdown() {
        for (int h = 0; h < 2; ++h) {
            if (hand[h] != XR_NULL_HANDLE && pfnDestroyHand) pfnDestroyHand(hand[h]);
            if (aimSpace[h] != XR_NULL_HANDLE) xrDestroySpace(aimSpace[h]);
        }
        if (actionSet != XR_NULL_HANDLE) xrDestroyActionSet(actionSet);
        for (auto& e : eyes) if (e.swapchain != XR_NULL_HANDLE) xrDestroySwapchain(e.swapchain);
        if (localSpace != XR_NULL_HANDLE) xrDestroySpace(localSpace);
        if (session != XR_NULL_HANDLE) xrDestroySession(session);
        if (instance != XR_NULL_HANDLE) xrDestroyInstance(instance);
        audio.shutdown();
        gl.shutdown();
        if (eglDisplay != EGL_NO_DISPLAY) {
            eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (eglSurface != EGL_NO_SURFACE) eglDestroySurface(eglDisplay, eglSurface);
            if (eglContext != EGL_NO_CONTEXT) eglDestroyContext(eglDisplay, eglContext);
            eglTerminate(eglDisplay);
        }
    }
};

void onAppCmd(android_app* a, int32_t cmd) {
    App* s = static_cast<App*>(a->userData);
    if (cmd == APP_CMD_RESUME) s->resumed = true;
    else if (cmd == APP_CMD_PAUSE) s->resumed = false;
}

}  // namespace

void android_main(android_app* app) {
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);

    App a;
    a.app = app;
    a.env = env;
    app->userData = &a;
    app->onAppCmd = onAppCmd;

    bool ok = a.init();
    if (!ok) { LOGE("Initialization failed - closing"); ANativeActivity_finish(app->activity); }

    bool finishing = !ok;
    while (!app->destroyRequested) {
        for (;;) {
            // Block only while paused with no running session; otherwise poll without waiting.
            const int timeout = (!a.resumed && !a.sessionRunning && !app->destroyRequested) ? -1 : 0;
            int events = 0;
            android_poll_source* src = nullptr;
            if (ALooper_pollAll(timeout, nullptr, &events, (void**)&src) < 0) break;
            if (src) src->process(app, src);
            if (app->destroyRequested) break;
        }
        if (!a.ready || app->destroyRequested) continue;
        a.pollEvents();
        if (a.exitRequested && !finishing) { finishing = true; ANativeActivity_finish(app->activity); }
        a.ensureHandTrackers();
        if (a.sessionRunning) a.frame();
    }

    if (a.ready) a.shutdown();
    app->activity->vm->DetachCurrentThread();
}
