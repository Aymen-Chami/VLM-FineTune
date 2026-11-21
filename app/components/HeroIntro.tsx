"use client";

import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { FloatingCursor } from "@/components/FloatingCursor";
import { site } from "@/lib/config";

const fadeUp = (delay: number) => ({
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] as const },
});

export function HeroIntro() {
  const [leftCursor, rightCursor] = site.cursors;
  return (
    <>
      <motion.a
        href={site.badge.href}
        {...fadeUp(0)}
        className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 py-1 pl-1 pr-3 text-[13px] text-white/90 backdrop-blur-md transition-colors hover:bg-white/15 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
      >
        <span className="rounded-full bg-white px-2 py-0.5 text-[11px] font-semibold tracking-wide text-electric">{site.badge.label}</span>
        {site.badge.text}
        <ArrowRight className="size-3.5" aria-hidden />
      </motion.a>

      <div className="relative mt-8">
        <FloatingCursor name={leftCursor.name} side="left" className="-left-20 top-[16%] hidden lg:flex" />
        <FloatingCursor name={rightCursor.name} side="right" className="-right-20 top-[96%] hidden lg:flex" />
        <motion.h1
          {...fadeUp(0.08)}
          className="text-[clamp(2.6rem,7.4vw,4.75rem)] font-semibold leading-[1.04] tracking-[-0.035em] text-white [text-shadow:0_2px_30px_rgba(2,6,26,0.25)]"
        >
          {site.headline.line1}
          <br />
          <em className="font-semibold">{site.headline.line2}</em>
        </motion.h1>
      </div>

      <motion.p {...fadeUp(0.16)} className="mt-6 max-w-xl text-[17px] leading-relaxed text-white/80">
        {site.subheadline[0]}
        <br className="hidden sm:block" /> {site.subheadline[1]}
      </motion.p>
    </>
  );
}
