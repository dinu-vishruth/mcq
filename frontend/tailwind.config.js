/** @type {import('tailwindcss').Config} */
// Tokens mirror the existing static/css/style.css :root variables so React
// screens are visually identical to the server-rendered pages during the
// incremental migration.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: "#09090b",
        elev: "#0e0e11",
        card: "#131316",
        "card-2": "#17171b",
        "card-hover": "#1b1b20",
        inset: "#0b0b0d",
        text: { DEFAULT: "#f4f4f5", 2: "#a1a1aa", 3: "#71717a" },
        accent: {
          DEFAULT: "#3b82f6",
          2: "#60a5fa",
          strong: "#2563eb",
        },
        success: "#10b981",
        warning: "#f59e0b",
        danger: "#ef4444",
        violet: "#8b5cf6",
      },
      borderColor: {
        DEFAULT: "rgba(255,255,255,0.07)",
        2: "rgba(255,255,255,0.11)",
        strong: "rgba(255,255,255,0.16)",
      },
      borderRadius: {
        xs: "8px",
        sm: "11px",
        md: "14px",
        lg: "18px",
        xl: "24px",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        display: ["Sora", "Inter", "sans-serif"],
      },
      boxShadow: {
        sm: "0 1px 2px rgba(0,0,0,0.4)",
        md: "0 4px 16px rgba(0,0,0,0.35)",
        lg: "0 16px 48px rgba(0,0,0,0.5)",
        glow: "0 8px 32px rgba(59,130,246,0.22)",
      },
      transitionTimingFunction: {
        smooth: "cubic-bezier(0.22,0.61,0.36,1)",
        out: "cubic-bezier(0.16,1,0.3,1)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.4s cubic-bezier(0.16,1,0.3,1) both",
      },
    },
  },
  plugins: [],
};
