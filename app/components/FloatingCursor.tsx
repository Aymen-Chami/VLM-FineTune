"use client";

import { motion } from "framer-motion";
import { MousePointer2, Navigation } from "lucide-react";

export function FloatingCursor({ name, side, className = "" }: { name: string; side: "left" | "right"; className?: string }) {
  const left = side === "left";
  return (
    <motion.div
      aria-hidden
      animate={{ y: [0, -8, 0], x: [0, left ? 5 : -5, 0] }}
      transition={{ duration: left ? 5 : 6, repeat: Infinity, ease: "easeInOut", delay: left ? 0 : 1.2 }}
      className={`pointer-events-none absolute flex flex-col gap-0.5 ${left ? "items-end" : "items-start"} ${className}`}
    >
      {left ? (
        <Navigation className="mr-0 size-5 rotate-0 text-sky-200" strokeWidth={1.75} />
      ) : (
        <MousePointer2 className="size-5 -rotate-12 fill-sky-200 text-sky-200" strokeWidth={1.5} />
      )}
      <span className={`rounded-md border border-white/25 bg-[#1e5fe0]/45 px-1.5 py-0.5 text-[11px] font-medium text-white/90 backdrop-blur-sm ${left ? "mr-4" : "ml-4"}`}>
        {name}
      </span>
    </motion.div>
  );
}
