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
        bg: "#f7f6f2",
        elev: "#fcfbf8",
        card: "#ffffff",
        "card-2": "#f3f2ed",
        "card-hover": "#f8f9fc",
        inset: "#f0f1ed",
        line: { DEFAULT: "#e2e4df", 2: "#d7dcd5", strong: "#bdc7c0" },
        text: { DEFAULT: "#202d3b", 2: "#526070", 3: "#687381" },
        accent: {
          DEFAULT: "#2855d9",
          2: "#2855d9",
          strong: "#2045b4",
        },
        success: "#16704a",
        warning: "#9a5b0b",
        danger: "#bc3644",
        violet: "#6e56a4",
      },
      borderColor: {
        DEFAULT: "#e2e4df",
        2: "#d7dcd5",
        strong: "#bdc7c0",
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
        editorial: ["Iowan Old Style", "Palatino Linotype", "Book Antiqua", "Georgia", "serif"],
      },
      boxShadow: {
        sm: "0 2px 4px rgba(32,45,59,0.025)",
        md: "0 8px 24px rgba(32,45,59,0.05)",
        lg: "0 18px 60px rgba(32,45,59,0.08)",
        glow: "0 4px 12px rgba(40,85,217,0.14)",
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
