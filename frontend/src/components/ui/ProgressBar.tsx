import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

export function ProgressBar({ pct, tone = "accent", className }: { pct: number; tone?: "accent" | "danger" | "success"; className?: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const fill = {
    accent: "bg-accent",
    danger: "bg-gradient-to-r from-danger to-[#f87171]",
    success: "bg-success",
  }[tone];
  return (
    <div className={cn("h-2 rounded-full bg-white/[0.06] overflow-hidden", className)}>
      <motion.div
        className={cn("h-full rounded-full", fill)}
        initial={{ width: 0 }}
        animate={{ width: `${clamped}%` }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
      />
    </div>
  );
}
