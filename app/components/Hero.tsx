import { AuroraBackground } from "@/components/AuroraBackground";
import { HeroIntro } from "@/components/HeroIntro";
import { PromptBox } from "@/components/PromptBox";

export function Hero() {
  return (
    <section className="relative isolate min-h-svh overflow-hidden">
      <AuroraBackground />
      <div className="mx-auto flex max-w-6xl flex-col items-center px-4 pb-24 pt-16 text-center sm:pt-20">
        <HeroIntro />
        <div className="mt-12 w-full">
          <PromptBox />
        </div>
      </div>
    </section>
  );
}
