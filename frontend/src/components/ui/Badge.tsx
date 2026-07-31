import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

type Tone = "accent" | "success" | "warning" | "danger" | "neutral";

const TONE: Record<Tone, string> = {
  accent: "bg-accent/12 text-accent-2 border-accent/20",
  success: "bg-success/12 text-success border-success/20",
  warning: "bg-warning/12 text-warning border-warning/20",
  danger: "bg-danger/12 text-danger border-danger/20",
  neutral: "bg-white/5 text-text-2 border-white/10",
};

export function Badge({ children, tone = "neutral", className }: { children: ReactNode; tone?: Tone; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium border [&_svg]:w-3.5 [&_svg]:h-3.5",
        TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
