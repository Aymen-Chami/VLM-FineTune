import { demo, models, templates } from "@/lib/config";
import { examples, metrics } from "@/lib/data";
import type { InferResponse, ModelId, ModelOutput } from "@/lib/types";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function error(message: string, status = 400) {
  return Response.json({ error: message }, { status });
}

function isOutput(x: unknown): x is ModelOutput {
  return typeof x === "object" && x !== null && typeof (x as ModelOutput).answer === "string";
}

export async function POST(request: Request) {
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return error("Expected multipart form data.");
  }

  const image = form.get("image");
  const question = String(form.get("question") ?? "").trim();
  const model = String(form.get("model") ?? "") as ModelId;
  const template = templates.find((t) => t.id === form.get("template"));
  const sampleId = String(form.get("sampleId") ?? "");
  const option = models.find((m) => m.id === model);

  if (!(image instanceof File) || image.size === 0) return error("Attach an image first.");
  if (!image.type.startsWith("image/")) return error("The file must be an image.");
  if (image.size > demo.maxUploadMB * 1024 * 1024) return error(`Images must be under ${demo.maxUploadMB} MB.`);
  if (!option) return error("Unknown model.");
  if (!template) return error("Unknown prompt template.");
  if (template.needsQuestion && !question) return error("Type a question.");
  if (question.length > demo.maxQuestionChars) return error(`Questions are limited to ${demo.maxQuestionChars} characters.`);

  const prompt = template.prompt.replace("{question}", question);

  // Real model server: POST multipart {image, prompt, question, template, model}
  // and expect {"base": {"answer", "latencyMs"}, "finetuned": {"answer", "latencyMs"}}.
  const upstream = process.env.INFERENCE_URL;
  if (upstream) {
    const body = new FormData();
    body.set("image", image);
    body.set("prompt", prompt);
    body.set("question", question);
    body.set("template", template.id);
    body.set("model", model);
    let res: Response;
    try {
      res = await fetch(upstream, { method: "POST", body });
    } catch {
      return error("The inference server is unreachable.", 502);
    }
    if (!res.ok) return error(`The inference server returned ${res.status}.`, 502);
    const data = await res.json();
    if (!isOutput(data?.base) || !isOutput(data?.finetuned)) return error("Unexpected response from the inference server.", 502);
    return Response.json({ source: "live", model, base: data.base, finetuned: data.finetuned } satisfies InferResponse);
  }

  const { min, max } = demo.mockDelayMs;
  await sleep(min + Math.random() * (max - min));

  const latency = metrics.families[model];
  const baseMs = Math.round(latency.base.latencyS * 1000);
  const ftMs = Math.round(latency.finetuned.latencyS * 1000);

  // Sample image + its original question: replay the outputs recorded during evaluation.
  const example = examples.find((e) => e.id === sampleId);
  if (example && template.id === "vqa" && example.question === question) {
    const p = example.predictions[model];
    return Response.json({
      source: "recorded",
      model,
      gold: example.answers,
      metric: example.dataset === "docvqa" ? "ANLS" : "Relaxed accuracy",
      base: { answer: p.base.answer, score: p.base.score, latencyMs: baseMs },
      finetuned: { answer: p.finetuned.answer, score: p.finetuned.score, latencyMs: ftMs },
    } satisfies InferResponse);
  }

  return Response.json({
    source: "mock",
    model,
    base: { answer: `[mock] ${option.label} zero-shot answer`, latencyMs: baseMs },
    finetuned: { answer: `[mock] ${option.label} fine-tuned answer`, latencyMs: ftMs },
  } satisfies InferResponse);
}
