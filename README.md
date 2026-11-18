# TensorView VR

**Spatial ONNX graph explorer and static compute analyzer for Meta Quest.** Load an ONNX model and walk through its
computation graph in 3D: every layer is a block you can point at and pinch to inspect its shapes,
parameter count, FLOPs and activation memory — with your hands, no controllers needed.

| Build | Where | Run it |
|---|---|---|
| **WebXR / IWSDK (primary, link-based)** | source `web/` → production build committed in `docs/` | Open the GitHub Pages link in the Meta Quest Browser → **ENTER VR**. Locally: `./serve.sh`. Rebuild: `./build_web.sh` |
| Native OpenXR (Quest APK) | `app/` (C++17, OpenXR + GLES3) | `./build_quest.sh` (needs Android SDK/NDK) |

**Live demo:** `https://migmargil.github.io/tensorview-vr/` (create the public repository `migmargil/tensorview-vr` and enable Pages: `main` branch, `/docs` folder)

## Features

| Area | What it does |
|---|---|
| **ONNX parsing** | Dependency-free protobuf reader (JS and C++ implementations that agree): nodes, initializers, shapes. Parameter counts, estimated FLOPs (Conv / depthwise / Gemm / MatMul / elementwise) and estimated FP32 activation size per layer. These are static estimates, not measured inference latency, peak live memory, GPU utilization, or runtime profiling. |
| **Three built-in architectures** | ResNet-style CNN (residual skips), MobileNet-style (depthwise-separable + inverted residuals), tiny Transformer (parallel Q/K/V, attention, FFN). Switch live with in-air buttons. Your own model: drop an `.onnx` on the desktop page or use `?model=<url>`. |
| **Spatial layout** | Layered DAG folded into snake rows; branches split in depth. Block size = compute cost; color = operator type, or a per-model **heat map** (button *COLOR*). Animated pulses show data flow. |
| **Inspector + hotspots** | Pinch a layer: shapes, params, FLOPs, memory, each with its **share of the whole model**. Nothing selected: the top-3 compute hotspots. |
| **Accessibility** | High-contrast mode (button *CONTRAST* / key C): pure black background, white outlines, bright text and edges. |
| **Hands-first input** | WebXR (via IWSDK) target ray (Quest's far-field hand pointer) + pinch to select; pinch empty space and drag to move the graph; **two hands to zoom**; tracked hand joints are drawn. Controllers work as a fallback. |
| **Guided first minutes** | 5-step onboarding with a progress bar that advances only when you perform each gesture. |
| **Sound design** | Procedural UI sounds (hover, select, grab, release, step) synthesized at runtime — no audio assets. |
| **Desktop preview** | Orbit / click preview with keyboard shortcuts (1/2/3 models, H heat, R reset). |

## Verification

| Check | How |
|---|---|
| ONNX parser cross-check against native C++ parser | `node tests/web_test.mjs` (totals match for 3 models; hand-calculated conv/gemm values) |
| Automated XR interaction test | `cd tests/e2e && npm install && npm test` — intended to exercise an IWER-emulated XR session. Not executed in this review environment; do not claim this test passed unless you run it successfully. |
| Native build | Experimental OpenXR/C++ implementation; not validated on a physical headset in this submission review. |

## Build the native Quest APK (optional)

Requirements: JDK 17, Gradle 8.4, Android SDK (platform 34, build-tools 34.0.0), NDK 25.2.9519653, CMake 3.22.1. (`build_quest.sh` installs the SDK platform for you if `sdkmanager` is on PATH.)

```bash
./build_quest.sh                # creates a signing key if missing, generates the sample model if missing, runs gradle assembleRelease
# -> app/build/outputs/apk/release/app-release.apk
adb install -r app/build/outputs/apk/release/app-release.apk
```

## Test the parser on your PC

```bash
python3 -m venv venv && ./venv/bin/pip install onnx numpy
./venv/bin/python tools/make_model.py /tmp/models      # writes the 3 sample models
g++ -std=c++17 tests/test_onnx.cpp app/src/main/cpp/onnx_model.cpp -Iapp/src/main/cpp -o /tmp/test_onnx
/tmp/test_onnx /tmp/models/tiny_transformer.onnx
```

## Code map

**WebXR (`web/src/`, built to `docs/`):** `main.js` (IWSDK `World.create` + system), `app.js` (scene, pointers, gestures, HUD wiring), `onnx.js` (parser + layout), `logic.js` (pure gesture/onboarding logic), `hud.js` (canvas panels), `audio.js`.

**Native (`app/src/main/cpp`):**

| File | Role |
|---|---|
| `xr_app.cpp` | `android_main`, OpenXR instance/session/swapchains, hand tracking, actions, frame loop, scene + HUD |
| `audio.cpp/.h` | AAudio procedural sound effects |
| `gl_batch.cpp/.h` | GLES3 batch renderer: solid boxes, beams, bitmap text (single shader + atlas) |
| `onnx_model.cpp/.h` | ONNX protobuf parser, cost model, graph layout |
| `tv_math.h` | Vectors, quaternions, OpenXR FOV projection, view matrix |
| `font5x7.h` | Embedded 5x7 bitmap font |

## Validation limits

The bundled ONNX files are synthetic demonstration architectures generated for this project; they are not trained benchmark models. The WebXR application has not been independently verified on a physical Meta Quest in this review. The screenshots and accompanying draft video are desktop-preview captures, not proof of a live immersive session.
