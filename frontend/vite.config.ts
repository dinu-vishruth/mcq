import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

// The Flask app serves everything from /static. We build the React bundle into
// static/dist with STABLE filenames (no content hash) so Jinja templates can
// reference /static/dist/app.js and /static/dist/app.css directly. Vercel only
// runs @vercel/python at deploy time (no Node step), so these built files are
// committed to the repo and shipped as-is.
export default defineConfig({
  plugins: [react()],
  base: "/static/dist/",
  resolve: {
    alias: { "@": resolve(__dirname, "src") },
  },
  build: {
    outDir: resolve(__dirname, "../static/dist"),
    emptyOutDir: true,
    manifest: false,
    rollupOptions: {
      input: resolve(__dirname, "src/main.tsx"),
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "chunk-[name].js",
        assetFileNames: (info) => {
          if (info.name && info.name.endsWith(".css")) return "app.css";
          return "assets/[name][extname]";
        },
      },
    },
  },
});
