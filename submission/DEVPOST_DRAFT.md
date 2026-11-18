# TensorView VR — final Devpost submission copy

Use the following values in the Devpost form. Create the GitHub repository with the exact name `tensorview-vr` under `MigMarGil` so the project URL below matches.

## Form fields

- **Submission name:** TensorView VR
- **Tagline:** Explore ONNX neural networks as interactive 3D graphs. Inspect layers, residual paths and estimated compute hotspots with hands-first spatial interaction.
- **Track:** Productivity
- **Division:** Adapted / Significantly Updated Experience
- **Project URL:** https://migmargil.github.io/tensorview-vr/
- **Built with:** Immersive Web SDK (IWSDK), WebXR, Three.js, JavaScript, ONNX protobuf, C++17, OpenXR, WebAudio
- **Target launch date:** 18 November 2026 (web demo; subject to successful public deployment)
- **Team:** Miguel Martín Gil — miguelmartingil28@gmail.com — Solo developer
- **Video URL:** Paste the public YouTube URL after uploading a compliant Quest / XR Simulator recording. The supplied `TensorView_VR_Demo_DRAFT.mp4` is an edited desktop-preview walkthrough and does NOT satisfy the competition's Quest/emulator-footage requirement.

## Project description (copy into Devpost)

Neural networks are usually explored through flat diagrams, layer tables and profiler dashboards. TensorView VR explores a different approach: turning an ONNX computation graph into a spatial structure that can be inspected as a 3D environment.

The application parses an ONNX model and lays out its operators as connected blocks. The graph makes sequential flow, branches and residual connections visible in space. Selecting a layer opens an inspector with its operator type, input and output shapes, parameter count, estimated floating-point operations and estimated FP32 activation size. A heat-map view and a ranked hotspot panel help identify operators with the largest estimated compute cost. These values are static estimates derived from graph metadata; they are not measured inference latency, GPU performance or peak runtime memory.

Three bundled demonstration architectures show different graph patterns: a ResNet-style convolutional network with residual paths, a MobileNet-style network with depthwise-separable operations, and a small Transformer with parallel query, key and value branches. Users can switch between them, inspect individual operators, reposition the graph and change its visual contrast. The sample ONNX files are synthetic architecture demonstrations rather than trained benchmark models.

The project is built around the Immersive Web SDK (IWSDK), Three.js and WebXR. The intended headset workflow is hands-first: point at an operator and pinch to inspect it; pinch empty space and drag to move the graph; use two hands to zoom. Controller input is retained as a fallback, and a desktop preview makes the graph available without a headset. A high-contrast mode supports clearer reading of graph elements and panels.

The ONNX reader is implemented directly over the protobuf wire format without requiring an ONNX runtime dependency. A separate C++17 parser provides a native implementation for cross-checking model totals. Automated parser and interaction-logic tests are included in the repository.

The original project began as a desktop graph-visualization prototype. During the competition period it was significantly expanded into an IWSDK/WebXR experience, with ONNX parsing and static cost analysis, spatial graph layout, hands-first interaction logic, multiple architecture demonstrations, compute heat maps, accessibility controls and guided onboarding.

Next steps include validating the full experience on physical Quest hardware, measuring frame rate and memory on-device, adding runtime trace ingestion for measured profiling, improving custom-model compatibility and adding side-by-side architecture comparison.

## Hand interactions (optional field)

TensorView VR is designed around hands-first interaction. In the headset, the intended workflow uses the WebXR input target ray for pointing and pinch gestures for selection and graph manipulation. Pinching a graph block opens its inspector; pinching empty space and moving the hand repositions the graph; two hands control scale. Controllers are supported as a fallback. The interaction code is implemented, but this submission package does not claim a physical Quest validation run.

## Adapted / Significantly Updated — new features

The project began as a desktop GLFW/OpenGL prototype with a mouse-driven simulated pointer and a hard-coded graph. During the competition window, it was significantly updated with: an IWSDK/WebXR application targeting Meta Quest Browser; WebXR hand/controller input pathways and pinch-based interaction logic; a dependency-free ONNX protobuf parser with per-layer parameter and static compute estimates; three model architecture demonstrations and model switching; spatial graph layout with residual/parallel branches; compute heat-map and hotspot views; high-contrast accessibility mode; guided onboarding; and procedural UI audio.

## Testing instructions

1. Open the public project link in Meta Quest Browser.
2. Select **ENTER VR** when WebXR is available.
3. Follow the onboarding prompts: point, pinch to inspect, pinch empty space and move, then use both hands to zoom.
4. Switch between ResNet, MobileNet and Transformer; inspect a block and enable the compute heat map / high contrast.
5. If no headset is available, use the desktop preview to inspect the bundled architectures.

**Known validation status:** parser and pure interaction-logic tests pass locally; the native C++ parser compiles and parses the three bundled models. The E2E IWER test and a physical Quest session were not completed in this review environment. Do not describe them as verified.
