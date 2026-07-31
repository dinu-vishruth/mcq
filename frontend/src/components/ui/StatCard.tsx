import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { Card } from "./Card";
import { cn } from "@/lib/cn";

interface Props {
  value: ReactNode;
  label: string;
  icon: ReactNode;
  tone?: "accent" | "emerald" | "amber" | "violet";
  index?: number;
}

const TONE: Record<NonNullable<Props["tone"]>, string> = {
  accent: "text-accent bg-accent/10",
  emerald: "text-success bg-success/10",
  amber: "text-warning bg-warning/10",
  violet: "text-violet bg-violet/10",
};

export function StatCard({ value, label, icon, tone = "accent", index = 0 }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <Card hover>
        <div className="flex items-start justify-between">
          <div>
            <div className="text-[1.7rem] font-display font-bold leading-none tracking-tight">
              {value}
            </div>
            <div className="text-text-3 text-sm mt-1.5">{label}</div>
          </div>
          <span className={cn("grid place-items-center w-10 h-10 rounded-sm [&_svg]:w-5 [&_svg]:h-5", TONE[tone])}>
            {icon}
          </span>
        </div>
      </Card>
    </motion.div>
  );
}
