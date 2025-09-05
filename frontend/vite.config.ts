import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "",  // Set base to empty string for relative asset paths
  build: {
    outDir: "build"
  }
});
