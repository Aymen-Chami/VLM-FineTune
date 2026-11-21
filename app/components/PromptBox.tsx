"use client";

import { motion } from "framer-motion";
import { Boxes, ImageIcon, Images, Mic, Palette, Plus, SendHorizontal, X } from "lucide-react";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { ChipMenu } from "@/components/ChipMenu";
import { ResultPanel, type DemoState } from "@/components/ResultPanel";
import { defaultModel, demo, models, templates } from "@/lib/config";
import { examples } from "@/lib/data";
import type { InferResponse, ModelId, TemplateId } from "@/lib/types";

interface Attached {
  file: File;
  url: string;
  sampleId?: string;
}

interface Recognition {
  lang: string;
  interimResults: boolean;
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start(): void;
  stop(): void;
}

function speechCtor(): (new () => Recognition) | undefined {
  const w = window as unknown as Record<string, unknown>;
  return (w.SpeechRecognition ?? w.webkitSpeechRecognition) as (new () => Recognition) | undefined;
}

const noopSubscribe = () => () => {};

export function PromptBox() {
  const [model, setModel] = useState<ModelId>(defaultModel);
  const [templateId, setTemplateId] = useState<TemplateId>("vqa");
  const [attached, setAttached] = useState<Attached | null>(null);
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<DemoState>({ status: "idle" });
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [listening, setListening] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<Recognition | null>(null);
  const speechSupported = useSyncExternalStore(noopSubscribe, () => window.isSecureContext && !!speechCtor(), () => false);

  const option = models.find((m) => m.id === model)!;
  const template = templates.find((t) => t.id === templateId)!;
  const loading = state.status === "loading";

  useEffect(() => {
    return () => {
      if (attached?.url.startsWith("blob:")) URL.revokeObjectURL(attached.url);
    };
  }, [attached]);

  function attachFile(file: File) {
    if (!file.type.startsWith("image/")) return setNotice("That file isn't an image. Use PNG, JPG or WebP.");
    if (file.size > demo.maxUploadMB * 1024 * 1024) return setNotice(`Images must be under ${demo.maxUploadMB} MB.`);
    setNotice(null);
    setAttached({ file, url: URL.createObjectURL(file) });
  }

  async function pickSample(id: string) {
    const ex = examples.find((e) => e.id === id)!;
    try {
      const blob = await (await fetch(ex.image)).blob();
      setAttached({ file: new File([blob], `${ex.id}.jpg`, { type: blob.type || "image/jpeg" }), url: ex.image, sampleId: ex.id });
      setTemplateId("vqa");
      setQuestion(ex.question);
      setNotice(null);
      textRef.current?.focus();
    } catch {
      setNotice("Couldn't load that sample image.");
    }
  }

  async function submit() {
    if (loading) return;
    if (!attached) return setNotice("Add an image first: upload one, drop it here, or pick a sample from Reference.");
    if (template.needsQuestion && !question.trim()) {
      setNotice("Type a question about the image.");
      return textRef.current?.focus();
    }
    setNotice(null);
    setState({ status: "loading", model });

    const body = new FormData();
    body.set("image", attached.file);
    body.set("question", question.trim());
    body.set("model", model);
    body.set("template", templateId);
    if (attached.sampleId) body.set("sampleId", attached.sampleId);

    try {
      const res = await fetch("/api/infer", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `Request failed (${res.status}).`);
      setState({ status: "done", result: data as InferResponse });
    } catch (e) {
      setState({ status: "error", message: e instanceof Error ? e.message : "Request failed." });
    }
  }

  function toggleMic() {
    if (listening) return recognitionRef.current?.stop();
    const Ctor = speechCtor();
    if (!Ctor) return;
    const rec = new Ctor();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.onresult = (e) => {
      const text = Array.from(e.results).map((r) => r[0].transcript).join(" ");
      setQuestion((q) => (q ? `${q} ${text}` : text));
    };
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    recognitionRef.current = rec;
    setListening(true);
    rec.start();
  }

  return (
    <section id="demo" aria-label="Live demo" className="mx-auto w-full max-w-[640px] scroll-mt-24 text-left">
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files[0];
          if (file) attachFile(file);
        }}
        className={`relative rounded-3xl border bg-white/60 p-3 shadow-[0_24px_60px_-18px_rgba(10,77,255,0.45)] backdrop-blur-xl transition-colors focus-within:border-electric-light/60 ${
          dragging ? "border-electric-light ring-4 ring-electric-light/20" : "border-white/80"
        }`}
      >
        {dragging && (
          <div className="pointer-events-none absolute inset-0 z-10 grid place-items-center rounded-3xl bg-mist/80 text-sm font-medium text-electric">
            Drop the image to attach it
          </div>
        )}

        {attached && (
          <div className="mb-2 flex items-center gap-3 rounded-2xl bg-white/70 p-2">
            {/* eslint-disable-next-line @next/next/no-img-element -- previews of user uploads (blob: URLs) */}
            <img src={attached.url} alt="Attached image preview" className="size-14 rounded-xl border border-slate-200 bg-white object-cover" />
            <div className="min-w-0 flex-1 text-sm">
              <p className="truncate font-medium text-ink">{attached.sampleId ? `Sample · ${attached.sampleId}` : attached.file.name}</p>
              <p className="text-xs text-slate-500">{(attached.file.size / 1024).toFixed(0)} KB</p>
            </div>
            <button
              type="button"
              onClick={() => setAttached(null)}
              aria-label="Remove image"
              className="grid size-8 place-items-center rounded-full text-slate-500 hover:bg-mist hover:text-ink focus-visible:outline-2 focus-visible:outline-electric-light"
            >
              <X className="size-4" aria-hidden />
            </button>
          </div>
        )}

        <label htmlFor="demo-question" className="sr-only">
          Question about the image
        </label>
        <textarea
          id="demo-question"
          ref={textRef}
          rows={2}
          value={question}
          maxLength={demo.maxQuestionChars}
          placeholder={template.placeholder}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          onPaste={(e) => {
            const file = Array.from(e.clipboardData.files).find((f) => f.type.startsWith("image/"));
            if (file) {
              e.preventDefault();
              attachFile(file);
            }
          }}
          className="block max-h-40 min-h-12 w-full resize-none bg-transparent px-2 pt-1 text-[15px] text-ink [field-sizing:content] placeholder:text-slate-400 focus:outline-none"
        />

        <div className="mt-2 flex items-center gap-2">
          <div className="no-scrollbar flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto p-0.5">
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="sr-only"
              tabIndex={-1}
              aria-hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) attachFile(file);
                e.target.value = "";
              }}
            />
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              aria-label="Upload an image"
              className="grid size-7 shrink-0 place-items-center rounded-full border border-slate-200/80 bg-white/70 text-slate-700 transition-colors hover:bg-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-electric-light"
            >
              <Plus className="size-4" aria-hidden />
            </button>
            <ChipMenu
              ariaLabel="Model"
              label={option.label}
              icon={<Boxes className="size-3.5 text-slate-500" aria-hidden />}
              value={model}
              onSelect={setModel}
              items={models.map((m) => ({ value: m.id, label: m.label, description: m.detail }))}
            />
            <ChipMenu
              ariaLabel="Reference sample"
              label="Reference"
              icon={<Images className="size-3.5 text-slate-500" aria-hidden />}
              value={attached?.sampleId ?? null}
              onSelect={pickSample}
              width={320}
              items={examples.map((e) => ({
                value: e.id,
                label: e.question,
                description: `${e.dataset === "docvqa" ? "DocVQA" : "ChartQA"} · ${e.category}`,
                thumbnail: e.image,
              }))}
            />
            <ChipMenu
              ariaLabel="Prompt style"
              label={template.label === "VQA" ? "Style" : template.label}
              icon={<Palette className="size-3.5 text-slate-500" aria-hidden />}
              value={templateId}
              onSelect={setTemplateId}
              items={templates.map((t) => ({ value: t.id, label: t.label, description: t.needsQuestion ? "Ask a question" : t.prompt }))}
            />
          </div>

          <div className="flex shrink-0 items-center gap-1.5">
            <button
              type="button"
              onClick={toggleMic}
              disabled={!speechSupported}
              aria-pressed={listening}
              aria-label={speechSupported ? (listening ? "Stop dictation" : "Dictate question") : "Dictation needs HTTPS and Chrome or Edge"}
              title={speechSupported ? undefined : "Dictation needs HTTPS and Chrome or Edge"}
              className={`grid size-9 place-items-center rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-electric-light disabled:cursor-not-allowed disabled:opacity-40 ${
                listening ? "bg-electric-light/15 text-electric" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              <Mic className={`size-4 ${listening ? "motion-safe:animate-pulse" : ""}`} aria-hidden />
            </button>
            <button
              type="button"
              onClick={submit}
              disabled={loading}
              aria-label="Send"
              className="grid size-9 place-items-center rounded-full bg-electric-light text-white shadow-[0_6px_16px_-4px_rgba(30,123,255,0.7)] transition-transform hover:scale-105 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-electric disabled:opacity-60 motion-reduce:hover:scale-100"
            >
              <SendHorizontal className="size-4" aria-hidden />
            </button>
          </div>
        </div>
      </motion.div>

      <div aria-live="polite">
        {notice && (
          <p className="mt-3 flex items-center justify-center gap-2 text-sm text-slate-600">
            <ImageIcon className="size-4 text-electric-light" aria-hidden />
            {notice}
          </p>
        )}
      </div>

      <ResultPanel state={state} onRetry={submit} />
    </section>
  );
}
