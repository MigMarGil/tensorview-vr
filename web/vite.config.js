import { defineConfig } from 'vite';

// Builds the IWSDK app into ../docs (served by GitHub Pages: branch main, folder /docs).
export default defineConfig({
  build: { outDir: '../docs', emptyOutDir: true, target: 'esnext', sourcemap: false, chunkSizeWarningLimit: 4000 },
  esbuild: { target: 'esnext' },
  resolve: { dedupe: ['three', '@pmndrs/uikit', '@pmndrs/uikit-horizon', '@pmndrs/uikit-lucide'] },
  optimizeDeps: { exclude: ['@babylonjs/havok'], esbuildOptions: { target: 'esnext' } },
  publicDir: 'public',
  base: './',
});
