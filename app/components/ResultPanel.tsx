"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Check, CircleAlert, CircleDashed, RotateCw, Timer, X } from "lucide-react";
import { models } from "@/lib/config";
import type { InferResponse, ModelId, ModelOutput } from "@/lib/types";

export type DemoState =
  | { status: "idle" }
  | { status: "loading"; model: ModelId }
  | { status: "done"; result: InferResponse }
  | { status: "error"; message: string };

const sourceNote: Record<InferResponse["source"], string> = {
  recorded: "Recorded output from the evaluation run for this exact image and question. Latency is the average per question on an A10G.",
  mock: "Mock response: no model server is connected. Set INFERENCE_URL to run the real models (see README).",
  live: "Live output from the model server.",
};

const sourceLabel: Record<InferResponse["source"], string> = {
  recorded: "Recorded eval",
  mock: "Mock",
  live: "Live",
};

export function ResultPanel({ state, onRetry }: { state: DemoState; onRetry: () => void }) {
  return (
    <div aria-live="polite" aria-busy={state.status === "loading"}>
      <AnimatePresence mode="wait">
        {state.status !== "idle" && (
          <motion.div
            key={state.status}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="mt-4 rounded-3xl border border-white/80 bg-white/70 p-4 text-left shadow-[0_16px_40px_-20px_rgba(10,77,255,0.35)] backdrop-blur-xl"
          >
            {state.status === "loading" && <Loading model={state.model} />}
            {state.status === "error" && (
              <div className="flex flex-wrap items-center gap-3 text-sm text-ink">
                <CircleAlert className="size-5 text-rose-600" aria-hidden />
                <span className="flex-1">{state.message}</span>
                <button
                  type="button"
                  onClick={onRetry}
                  className="inline-flex items-center gap-1.5 rounded-full bg-ink px-3 py-1.5 text-xs font-medium text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-electric-light"
                >
                  <RotateCw className="size-3.5" aria-hidden /> Retry
                </button>
              </div>
            )}
            {state.status === "done" && <Done result={state.result} />}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Loading({ model }: { model: ModelId }) {
  const label = models.find((m) => m.id === model)!.label;
  return (
    <>
      <p className="mb-3 text-sm text-slate-600">Running {label} (base and fine-tuned)…</p>
      <div className="grid gap-3 sm:grid-cols-2">
        {[0, 1].map((i) => (
          <div key={i} className="rounded-2xl border border-slate-100 bg-white/80 p-4">
            <div className="skeleton h-3 w-24 rounded-full motion-safe:animate-shimmer" />
            <div className="skeleton mt-4 h-6 w-3/4 rounded-lg motion-safe:animate-shimmer" />
            <div className="skeleton mt-3 h-3 w-16 rounded-full motion-safe:animate-shimmer" />
          </div>
        ))}
      </div>
    </>
  );
}

function Done({ result }: { result: InferResponse }) {
  const option = models.find((m) => m.id === result.model)!;
  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold text-ink">{option.label}</span>
        <span className="rounded-full bg-mist px-2 py-0.5 text-xs font-medium text-electric">{sourceLabel[result.source]}</span>
        {result.gold && (
          <span className="ml-auto text-slate-600">
            Gold answer: <span className="font-medium text-ink">{result.gold[0]}</span>
          </span>
        )}
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <Column title="Base model" subtitle={option.baseLabel} output={result.base} source={result.source} metric={result.metric} />
        <Column title="Fine-tuned model" subtitle={option.finetunedLabel} output={result.finetuned} source={result.source} metric={result.metric} highlight />
      </div>
      <p className="mt-3 text-xs text-slate-500">{sourceNote[result.source]}</p>
    </>
  );
}

function Column({
  title,
  subtitle,
  output,
  source,
  metric,
  highlight,
}: {
  title: string;
  subtitle: string;
  output: ModelOutput;
  source: InferResponse["source"];
  metric?: string;
  highlight?: boolean;
}) {
  return (
    <div className={`rounded-2xl border bg-white/90 p-4 ${highlight ? "border-electric-light/40" : "border-slate-100"}`}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
          <p className="text-xs text-slate-400">{subtitle}</p>
        </div>
        <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">
          <Timer className="size-3" aria-hidden />
          {source === "live" ? `${output.latencyMs} ms` : `~${output.latencyMs} ms avg`}
        </span>
      </div>
      <p className="mt-3 break-words text-lg font-semibold text-ink">{output.answer || "(empty answer)"}</p>
      {output.score !== undefined && <Score score={output.score} metric={metric} />}
    </div>
  );
}

function Score({ score, metric }: { score: number; metric?: string }) {
  const [Icon, label, tone] =
    score >= 1 ? [Check, "Correct", "text-emerald-700"] : score > 0 ? [CircleDashed, "Partly correct", "text-amber-700"] : [X, "Wrong", "text-rose-700"];
  return (
    <p className={`mt-2 inline-flex items-center gap-1 text-xs font-medium ${tone}`}>
      <Icon className="size-3.5" aria-hidden />
      {label}
      <span className="text-slate-500">
        · {metric} {score.toFixed(2)}
      </span>
    </p>
  );
}
