// All page copy and demo settings live here.
import type { ModelId, TemplateId } from "./types";

export const site = {
  title: "VLM Fine-Tune — Florence-2 & PaliGemma 2 on DocVQA and ChartQA",
  description:
    "Fine-tuning two small open vision-language models for document and chart question answering, with before/after evaluation and a live demo.",
  badge: { label: "NEW", text: "Try the fine-tuned checkpoint", href: "#demo" },
  headline: { line1: "Vision-language,", line2: "fine-tuned for your domain" },
  subheadline: [
    "One workspace for training runs,",
    "side-by-side evaluation, and full experiment tracking.",
  ],
  cursors: [
    { name: "v0rtex_88", side: "left" as const },
    { name: "Michael Brooks", side: "right" as const },
  ],
};

export interface ModelOption {
  id: ModelId;
  label: string;
  detail: string;
  baseLabel: string;
  finetunedLabel: string;
}

export const models: ModelOption[] = [
  {
    id: "paligemma2",
    label: "PaliGemma 2",
    detail: "3B · LoRA",
    baseLabel: "pt-448, zero-shot",
    finetunedLabel: "LoRA r16, step 4000",
  },
  {
    id: "florence2",
    label: "Florence-2",
    detail: "base · 0.23B",
    baseLabel: "base-ft, zero-shot",
    finetunedLabel: "full fine-tune, step 3000",
  },
];

export const defaultModel: ModelId = "paligemma2";

export interface PromptTemplate {
  id: TemplateId;
  label: string;
  // {question} is replaced by the user's text
  prompt: string;
  needsQuestion: boolean;
  placeholder: string;
}

export const templates: PromptTemplate[] = [
  {
    id: "vqa",
    label: "VQA",
    prompt: "{question}",
    needsQuestion: true,
    placeholder: "Upload an image and ask a question…",
  },
  {
    id: "caption",
    label: "Captioning",
    prompt: "Describe this image in one sentence.",
    needsQuestion: false,
    placeholder: "Upload an image to caption it (optional note)…",
  },
  {
    id: "classify",
    label: "Classification",
    prompt: "Is this a chart or a document? Answer with one word.",
    needsQuestion: false,
    placeholder: "Upload an image to classify it…",
  },
];

export const demo = {
  maxUploadMB: 10,
  maxQuestionChars: 300,
  // simulated round-trip for the mock API (ms)
  mockDelayMs: { min: 700, max: 1300 },
};
