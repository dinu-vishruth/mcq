import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

// The Flask app serves everything from /static. We build the React bundle into
// static/dist with content hashes so a cached screen cannot outlive its backend
// contract. Jinja resolves the entry and stylesheet through the manifest. Vercel only
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
    manifest: "manifest.json",
    rollupOptions: {
      input: resolve(__dirname, "src/main.tsx"),
      output: {
        entryFileNames: "app-[hash].js",
        chunkFileNames: "chunk-[name]-[hash].js",
        assetFileNames: (info) => {
          if (info.name && info.name.endsWith(".css")) return "app-[hash].css";
          return "assets/[name]-[hash][extname]";
        },
      },
    },
  },
});
