import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The site is served from a folder, sohanbhat.me/signnet/, so the deploy
// builds with SITE_BASE=/signnet/. Without it, as in local development,
// everything is served from the root.
export default defineConfig({
  base: process.env.SITE_BASE ?? "/",
  plugins: [react()],
});
