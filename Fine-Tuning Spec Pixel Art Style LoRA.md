# Fine-Tuning Spec: Pixel Art Style LoRA

## Goal

Fine-tune a small, fast image model to reliably produce authentic grid-aligned pixel art in a chosen style, showing a visible quality jump over the base model — and show that the fine-tuned small model matches a bigger, more expensive reference model's quality at a fraction of the cost/latency.

## Model

Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`), standard 30-50 step sampling. Genuinely small (~860M UNet params, vs. SDXL's ~2.6B) rather than step-distilled — avoids the earlier SDXL-Lightning approach, where the 2-step distilled schedule left almost no tolerance for LoRA-induced bias (small, well-converged training losses still produced incoherent output, since a normal diffusion model can self-correct errors across many steps but a 2-step one can't). Standard multi-step sampling is a much more robust target for LoRA fine-tuning.

## Style

A niche 80s/90s console pixel art look (e.g. NES/SNES sprite aesthetic) — narrow enough that the base model's output stays generic and blurry.

## Data

20-50 real sprite/tile images in the target style, from public domain or CC-licensed sprite/tile packs.

Preprocessing: resize to training resolution with **nearest-neighbor** interpolation only (e.g. `Image.NEAREST` in PIL). Bicubic/bilinear resizing introduces the exact blur the correctness check rejects. Caption each image with a fixed trigger word (e.g. `"pxlnes style sprite"`) rather than auto-captioning — simpler and sufficient for a single-style LoRA.

## Baseline check

Generate images with the base model (2-step) on a fixed prompt set. Score with the correctness check below before training. If already good, stop.

## Correctness check

For each generated image, validate:

- **Edge sharpness** (replaces vague "grid alignment"): for each row/column, compute pixel-to-pixel brightness deltas. Bucket each delta as "flat" (~0) or "hard edge" (large jump). Score = % of transitions falling in neither bucket (i.e. soft/blurry transitions). Pixel art should score near 0%; blurry output scores high. Threshold empirically from the baseline run.
- Color count within target palette size (e.g. ≤16 unique colors after quantization)
- No smooth gradients (flat color blocks only) — captured by the edge-sharpness metric above
- Consistent pixel block size: estimate block size via autocorrelation of the sharpness signal (spacing between hard edges); check it's consistent across the image (low variance)

Score = % of images passing all checks. Supplement with a CLIP style-similarity score against reference images.

## Fine-tuning

- Base: Stable Diffusion 1.5
- Method: LoRA
- Libraries: `diffusers`, `peft`
- Standard random-timestep diffusion training (uniform over the full `num_train_timesteps` range) — no special scheduler alignment needed, since inference also uses standard multi-step sampling.

## Evaluation

Three-way comparison on the same fixed held-out prompts, in two phases:

**Phase 1 (free, self-hosted on the same Colab T4):**

1. SDXL base, 40 steps — "expensive" reference (~2.6B params, larger/slower)
2. SD1.5, 40 steps, no fine-tune — "cheap but bad"
3. SD1.5, 40 steps, fine-tuned LoRA — "cheap and good" (~860M params, smaller/faster than SDXL)

Generate all three at a fixed, matching resolution (512×512 — SD1.5's native resolution; see note below), same prompts, same seed where applicable.

**Phase 2 (optional, paid — only after Phase 1 pipeline works):** re-run prompt set 1 against a genuinely larger external model (e.g. FLUX.1 via a hosted API) to replace the "expensive" column with a true big-model reference and a real $ cost figure. Same scoring code, just a new image source.

**Reporting:**

- Metrics table: model | steps | approx. cost/latency per image | correctness score | CLIP style-similarity score
- Side-by-side image grid: rows = prompts, columns = [big model | small base | small fine-tuned]
- Cost-vs-quality scatter: x = cost or latency (log scale), y = correctness/CLIP score, one point per model. This is the single chart that makes the pitch — fine-tuned small model should land near the big model's quality at a fraction of its cost.

**Resolution note:** source sprites are 16×16, upscaled with nearest-neighbor to 512×512 (32px-wide "pixels") for both fine-tuning and all eval generations, so the block size matches the training data and looks like conventional pixel art across all three variants — including the SDXL reference, which also renders more coherently at 512×512 than it did at the previous, much-lower-than-native 256×256.

## Environment

Google Colab, free tier (T4 GPU), for manual/interactive runs. Kaggle Notebooks (free T4/P100 GPU quota), for unattended runs — see Automation below. Phase 2 (if pursued) adds a hosted API call, no extra compute environment needed.

## Automation

The full pipeline (`1_prepare_data.py` → `2_train_lora.py` → `3_generate.py` × 3 variants → `4_evaluate.py`) can be run unattended, triggered by a `git push`, instead of manually in a Colab session:

- **Why not GitHub Actions' own runners:** GitHub-hosted runners have no GPU. They can trigger and orchestrate a run, but can't execute the training/generation steps themselves.
- **Why not Colab:** the free tier has no supported API for headless/unattended execution — anything that drives it programmatically is browser automation, which is fragile.
- **Compute backend: Kaggle Notebooks.** Kaggle's `kernels push` API uploads and runs a script unattended on a free GPU (no live session needed) and exposes the run's output files for download afterward — built for exactly this.

Flow: `.github/workflows/run-pipeline.yml` triggers on push to `main` (or manually via `workflow_dispatch`). It pushes `kaggle/run_pipeline.py` as a Kaggle kernel (using `KAGGLE_USERNAME`/`KAGGLE_KEY` repo secrets), which `git clone`s this repo fresh inside the Kaggle GPU environment and runs the full pipeline end to end. The Action then polls the kernel's status, downloads its output once finished, and uploads it as a GitHub Actions artifact (metrics table, image grid, scatter plot, LoRA weights).

Tradeoff: Kaggle's free GPU quota is capped at 30 hours/week, and a single kernel run is capped around 9-12 hours — both far more than this pipeline needs, but worth knowing if it's ever scaled up.

## Budget

Phase 1: $0. Phase 2 (optional): a few dollars for ~10-20 API calls to the reference model, within the $0-10 target.
