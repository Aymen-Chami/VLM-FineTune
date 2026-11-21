export type ModelId = "paligemma2" | "florence2";
export type Stage = "base" | "finetuned";
export type TemplateId = "vqa" | "caption" | "classify";

export interface ModelOutput {
  answer: string;
  latencyMs: number;
  score?: number;
}

export interface InferResponse {
  source: "recorded" | "mock" | "live";
  model: ModelId;
  base: ModelOutput;
  finetuned: ModelOutput;
  gold?: string[];
  metric?: string;
}

export interface StageMetrics {
  docvqa: number;
  docvqaByType: Record<string, number>;
  chartqa: number;
  chartqaHuman: number;
  chartqaAugmented: number;
  latencyS: number;
}

export interface Metrics {
  families: Record<ModelId, Record<Stage, StageMetrics>>;
  docvqaTypeCounts: Record<string, number>;
}

export interface Example {
  id: string;
  dataset: "docvqa" | "chartqa";
  category: string;
  image: string;
  question: string;
  answers: string[];
  predictions: Record<ModelId, Record<Stage, { answer: string; score: number }>>;
}
