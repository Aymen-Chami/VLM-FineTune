# VLM Fine-Tune: web demo

Single-page chat demo: upload a document or chart, ask a question, and compare the **base** and **fine-tuned** answers of PaliGemma 2 or Florence-2 side by side.

Stack: Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4, Framer Motion.

## Deploy on the AWS instance (shareable link)

1. **Open the port.** In the EC2 console, open the instance's security group and add an inbound rule: *Custom TCP, port 3000, source 0.0.0.0/0* (or "My IP" while testing).
2. **Start it.** SSH in and run:

   ```bash
   cd <repo on the instance>        # wherever you cloned VLM-FineTune
   git pull
   bash scripts/serve_demo.sh
   ```

   The first run installs Node into `~/work`, builds the site, loads the models (about 1–2 minutes), and prints the link: `http://<public-ip>:3000`.
3. **Stop it** with `tmux kill-session -t demo`, which frees the GPU. Logs: `tmux attach -t demo`, or `~/work/logs/demo_*.log`.

How it fits together:

```
browser ──► :3000  Next.js (app/)  ──►  127.0.0.1:8000  scripts/demo_server.py  (GPU, not exposed)
                   POST /api/infer                       PaliGemma 2 + LoRA, Florence-2 base + fine-tuned
```

The model server loads the checkpoints selected during evaluation: `outputs/pg2_448/checkpoint-4000` and `outputs/florence2_base/checkpoint-3000`. To use the Hub copies instead, pass `--pg-adapter` / `--florence-ft` to `scripts/demo_server.py`.

Notes:
- The public IP changes every time the instance is stopped and started. Attach an Elastic IP if you need a fixed link.
- The instance bills about $1/h while the demo is up. Requests are processed one at a time on the GPU.
- Dictation (mic button) needs HTTPS, so it is disabled on the plain `http://` link.

## Run locally

Needs Node.js 20+.

```bash
cd app
npm install
npm run dev          # http://localhost:3000
```

Without a model server, `/api/infer` is a mock:

- A **Reference** sample asked with its original question returns the answers **recorded during evaluation** (real outputs, with scores).
- Anything else returns a clearly labelled `[mock]` answer.

To use a real model server, set `INFERENCE_URL` in `.env.local` (see `.env.example`).

## Code

```
app/page.tsx, layout.tsx        page = Hero only
app/api/infer/route.ts          validates the upload, then proxies to INFERENCE_URL (or mocks)
components/Hero.tsx             aurora background + headline + PromptBox
components/PromptBox.tsx        upload / drop / paste, model, sample, style, mic, send
components/ResultPanel.tsx      base vs fine-tuned answers with latency
lib/config.ts                   all copy and settings (headline, models, prompt styles, limits)
data/                           metrics.json + examples.json from scripts/export_web_data.py
public/samples/                 images for the Reference samples
```

### Model server contract

`/api/infer` forwards `multipart/form-data` to `INFERENCE_URL`:

| field      | value                                                  |
|------------|--------------------------------------------------------|
| `image`    | the uploaded image file                                |
| `prompt`   | final prompt (style template applied to the question)  |
| `question` | raw user text                                          |
| `template` | `vqa` / `caption` / `classify`                         |
| `model`    | `paligemma2` / `florence2`                             |

It expects JSON back:

```json
{ "base": { "answer": "us", "latencyMs": 190 }, "finetuned": { "answer": "Libya", "latencyMs": 225 } }
```
