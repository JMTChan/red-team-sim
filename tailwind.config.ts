import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
        display: ["var(--font-display)", "ui-monospace", "monospace"],
      },
      colors: {
        // Phosphor-terminal palette
        void: "#05070a",
        panel: "#0b0f16",
        edge: "#1b2330",
        cyan: "#22d3ee",
        amber: "#f59e0b",
        viper: "#34d399",
        blood: "#f43f5e",
        violet: "#a78bfa",
      },
      boxShadow: {
        glow: "0 0 24px -4px rgba(34,211,238,0.45)",
      },
    },
  },
  plugins: [],
};
export default config;
