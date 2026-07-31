import type { ReactNode } from "react";
import { motion } from "framer-motion";
import type { HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/cn";

interface Props extends Omit<HTMLMotionProps<"div">, "children"> {
  children: ReactNode;
  hover?: boolean;
  pad?: "sm" | "md" | "lg";
}

const PAD = { sm: "p-4", md: "p-5", lg: "p-7" };

export function Card({ children, hover, pad = "md", className, ...rest }: Props) {
  return (
    <motion.div
      className={cn(
        "bg-card border rounded-lg relative",
        "border-white/[0.07]",
        hover && "transition-colors duration-200 hover:bg-card-hover hover:border-white/[0.11]",
        PAD[pad],
        className as string,
      )}
      {...rest}
    >
      {children}
    </motion.div>
  );
}
