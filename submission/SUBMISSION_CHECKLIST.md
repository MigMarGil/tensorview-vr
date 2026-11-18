# Submission checklist — Meta VR Start Developer Competition 2026

Official deadline: 18 November 2026, 12:00 PM PST (21:00 CET). Official rules: https://start-developer-competition-26.devpost.com/rules

## Must be done by the entrant
- [ ] Confirm Meta VR Start membership and Devpost registration / eligibility.
- [ ] Create public GitHub repository `MigMarGil/tensorview-vr`. Upload the extracted contents of this ZIP to the repository root (do not upload only the ZIP archive).
- [ ] Enable GitHub Pages: branch `main`, folder `/docs`.
- [ ] Open https://migmargil.github.io/tensorview-vr/ and confirm the site loads.
- [ ] Test **ENTER VR** in Meta Quest Browser or the official/equivalent XR emulator.
- [ ] Record a real Quest / XR Simulator session under 3 minutes. The supplied MP4 is only a desktop-preview draft and is not sufficient for the official video requirement.
- [ ] Upload that video publicly to YouTube or Vimeo.
- [ ] Paste the final video URL into Devpost.
- [ ] Submit the form before the deadline and keep the project available through judging.

## Recommended Devpost choices
- Track: Productivity (hands-first productivity tool).
- Division: Adapted / Significantly Updated Experience, based on the existing desktop prototype described in the project history.
- Target launch: 18 November 2026, contingent on successful deployment.

## Validation status in this package
- `node tests/web_test.mjs`: passed in this environment.
- C++ parser: compiled with `g++ -std=c++17` and parsed all three bundled ONNX models.
- E2E IWER browser test: not verified here (dependency installation timed out).
- Physical Quest runtime: not verified here.
