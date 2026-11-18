# TensorView VR

**Explore neural networks in virtual reality.**

TensorView VR is a spatial ONNX graph explorer and static compute analyzer designed for Meta Quest. It transforms neural network computation graphs into interactive 3D environments where developers can explore model architecture, inspect individual layers, and understand computational complexity through spatial interaction.

Instead of looking at a traditional 2D graph, step inside your model.

Point at a layer and pinch to inspect its tensor shapes, parameter count, estimated FLOPs, and estimated activation memory — using your hands, without requiring controllers.

<details>
<summary>Project links</summary>

- **Live WebXR Demo:** https://migmargil.github.io/tensorview-vr/
- **Source Code:** https://github.com/MigMarGil/tensorview-vr
- **Author:** [@MigMarGil](https://github.com/MigMarGil)

</details>

---

## Why TensorView VR?

Neural networks are often represented as complex diagrams and tables. Understanding their topology, branching paths, and computational costs can become difficult as models grow.

TensorView VR explores a different approach: spatial visualization.

By turning computation graphs into interactive 3D structures, the project aims to make model architecture easier to navigate and computational hotspots easier to identify.

## Features

### 1. Interactive 3D Neural Network Visualization

- Converts ONNX computation graphs into interactive 3D layouts.
- Represents layers as spatial blocks connected by data-flow beams.
- Organizes complex graphs into layered, snake-like rows.
- Separates branching paths in depth.
- Uses block size to represent estimated computational cost.
- Uses operator-specific colors to distinguish layer types.
- Includes animated pulses to visualize data flow.

### 2. ONNX Model Analysis

TensorView VR includes a dependency-free JavaScript protobuf reader and a native C++17 parser.

The analyzer extracts:
- Graph nodes.
- Initializers.
- Tensor shapes.
- Parameter counts.
- Estimated FLOPs.
- Estimated FP32 activation memory per layer.

Supported computational estimates include:
- Conv
- Depthwise convolution
- Gemm
- MatMul
- Elementwise operations

**Important:** These are static estimates, not measured inference latency, peak live memory, GPU utilization, or runtime profiling results.

### 3. Built-in Demonstration Models

Three synthetic demonstration architectures are included:

| Model | Architecture |
|---|---|
| ResNet-style CNN | Convolutional network with residual connections |
| MobileNet-style | Depthwise-separable convolutions and inverted residuals |
| Tiny Transformer | Parallel Q/K/V projections, attention, and feed-forward network |

Switch between models using the in-air interface.

You can also load your own ONNX model through the desktop page using a local `.onnx` file or a URL:

```text
?model=<url>
```

### 4. Hands-First Interaction

Designed around spatial interaction rather than traditional desktop controls.

- WebXR hand interaction through IWSDK.
- Quest far-field hand pointer targeting.
- Pinch to select and inspect layers.
- Pinch empty space and drag to move the graph.
- Two-hand interaction for zooming.
- Visualized tracked hand joints.
- Controller input supported as a fallback.

### 5. Heat Map and Accessibility

- Per-model compute heat map.
- High-contrast mode with black background, bright text, and white outlines.
- Keyboard shortcut `C` to toggle high contrast.
- Keyboard shortcut `H` to toggle the heat map.

### 6. Guided Onboarding

A five-step interactive tutorial introduces the main gestures and controls.

The progress indicator advances when the corresponding interaction is performed.

### 7. Procedural Sound Design

Interface sounds are synthesized at runtime for:
- Hover
- Selection
- Grab
- Release
- Tutorial steps

No external audio assets are required.

### 8. Desktop Preview

The desktop version provides an orbit-and-click preview of the visualization.

Keyboard shortcuts:

| Key | Action |
|---|---|
| `1` | Switch to ResNet-style model |
| `2` | Switch to MobileNet-style model |
| `3` | Switch to Tiny Transformer |
| `H` | Toggle heat map |
| `R` | Reset the view |

---

## Technology Stack

| Component | Technology |
|---|---|
| Web application | JavaScript, Three.js |
| XR framework | IWSDK |
| Immersive interaction | WebXR |
| Native application | C++17 |
| Native XR | OpenXR |
| Graphics | OpenGL ES 3 |
| Model format | ONNX |
| Web build | Vite |
| Native build | Gradle, Android SDK and NDK |

---

## Project Architecture

```text
tensorview-vr/
├── app/                  # Native Android / OpenXR application
├── docs/                 # Built web application for GitHub Pages
├── submission/           # Competition submission materials
├── tests/                # Parser and interaction tests
├── tools/                # Model generation and utilities
├── web/                  # WebXR application source
├── README.md
├── build.gradle
├── build_quest.sh
├── build_web.sh
├── deploy.sh
├── gradle.properties
├── serve.sh
└── settings.gradle
```

### WebXR source map

The main web application is located in `web/src/` and builds into `docs/`.

| File | Responsibility |
|---|---|
| `main.js` | IWSDK world creation and system setup |
| `app.js` | Scene, pointers, gestures, and HUD integration |
| `onnx.js` | ONNX parser and graph layout |
| `logic.js` | Gesture and onboarding logic |
| `hud.js` | Canvas-based interface panels |
| `audio.js` | Procedural sound effects |

### Native C++ source map

Located in `app/src/main/cpp/`.

| File | Responsibility |
|---|---|
| `xr_app.cpp` | Android entry point, OpenXR instance/session/swapchains, hand tracking, actions, frame loop, scene, and HUD |
| `audio.cpp / audio.h` | AAudio procedural sound effects |
| `gl_batch.cpp / gl_batch.h` | GLES3 batch renderer for boxes, beams, and bitmap text |
| `onnx_model.cpp / onnx_model.h` | ONNX protobuf parser, cost model, and graph layout |
| `tv_math.h` | Vectors, quaternions, FOV projection, and view matrix |
| `font5x7.h` | Embedded 5×7 bitmap font |

---

## Getting Started

### Option 1: Try the Live Demo

Open the public application:

**https://migmargil.github.io/tensorview-vr/**

For immersive interaction, open the page in Meta Quest Browser on a compatible headset and enter VR.

The desktop version also provides a preview of the 3D visualization.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/MigMarGil/tensorview-vr.git
cd tensorview-vr
```

Start the local server:

```bash
./serve.sh
```

To rebuild the web application:

```bash
./build_web.sh
```

For immersive WebXR functionality, use a compatible browser and an appropriate secure context.

### Option 3: Build the Native Quest APK

The native implementation uses C++17, OpenXR, and OpenGL ES 3.

Requirements:
- JDK 17
- Gradle 8.4
- Android SDK
- Android platform 34
- Android Build Tools 34.0.0
- Android NDK 25.2.9519653
- CMake 3.22.1

Build:

```bash
./build_quest.sh
```

The script generates a signing key if missing, generates the sample model if needed, and runs the release Gradle build.

Expected output:

```text
app/build/outputs/apk/release/app-release.apk
```

Install on a connected Android device:

```bash
adb install -r app/build/outputs/apk/release/app-release.apk
```

**Note:** The native implementation is experimental and has not been validated on a physical Meta Quest as part of this submission review.

---

## Testing and Verification

### JavaScript ONNX parser

Run:

```bash
node tests/web_test.mjs
```

The parser tests cover three demonstration models and compare relevant totals against the native C++ parser, including hand-calculated convolution and Gemm cases.

### Native C++ parser

Generate the demonstration models:

```bash
python3 -m venv venv
./venv/bin/pip install onnx numpy
./venv/bin/python tools/make_model.py /tmp/models
```

Compile the parser test:

```bash
g++ -std=c++17 \
  tests/test_onnx.cpp \
  app/src/main/cpp/onnx_model.cpp \
  -Iapp/src/main/cpp \
  -o /tmp/test_onnx
```

Run:

```bash
/tmp/test_onnx /tmp/models/tiny_transformer.onnx
```

### Automated XR interaction tests

The repository includes an IWER-based XR interaction test setup.

To run:

```bash
cd tests/e2e
npm install
npm test
```

These tests were not executed during this submission review. Their presence should not be interpreted as evidence that the automated XR suite has passed.

---

## Validation and Limitations

TensorView VR is a developer visualization and static analysis tool.

The following limitations are important:

- Bundled ONNX files are synthetic demonstration architectures generated for this project. They are not trained benchmark models.
- FLOPs and activation memory are static estimates, not runtime measurements.
- The WebXR application has not been independently verified on a physical Meta Quest during this submission review.
- The screenshots and draft demonstration video are desktop-preview captures, not evidence of a live immersive Quest session.
- The automated IWER end-to-end test suite was not executed during this review.
- The native OpenXR implementation is experimental and has not been validated on a physical headset.
- The native implementation does not necessarily include every feature available in the web application.

These limitations distinguish implemented functionality from independently verified behavior.

---

## Competition

**Meta VR Start Developer Competition 2026**

TensorView VR was developed for the Meta VR Start Developer Competition 2026.

The project explores how virtual reality and spatial interaction can make neural network architecture easier to understand.

Its goal is to give developers a more intuitive way to inspect ONNX models, navigate graph topology, and identify computationally expensive layers.

---

## Author

**Miguel Martín Gil**

- GitHub: [@MigMarGil](https://github.com/MigMarGil)
- Repository: [MigMarGil/tensorview-vr](https://github.com/MigMarGil/tensorview-vr)
- Live Demo: https://migmargil.github.io/tensorview-vr/

---

## License

This project is distributed under the MIT License.

See [LICENCE](LICENCE) for details.
