# Publish TensorView VR on GitHub Pages

The ZIP is a complete repository snapshot. **Do not upload the ZIP archive as the only file in the repository**: GitHub will store it as an attachment and will not extract it, so GitHub Pages will not find `/docs/index.html`.

1. On GitHub, create a **public** repository named `tensorview-vr` under the account `MigMarGil`.
2. Extract this ZIP on your computer.
3. In the new empty repository, choose **Add file → Upload files** and upload the extracted repository contents (including the `docs/` folder and the root files). Commit to `main`.
4. Open **Settings → Pages**. Choose **Deploy from a branch**, branch `main`, folder `/docs`, then Save.
5. Wait for the deployment and open: https://migmargil.github.io/tensorview-vr/
6. Test that page in Meta Quest Browser or an XR emulator before entering the link in Devpost.

If the repository already exists, do not overwrite unrelated files without checking them first.
