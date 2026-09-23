import argparse
import json
import time
from pathlib import Path

from models import RESOLUTION, load_pipeline
from prompts import PROMPTS

VARIANTS = {
    "sdxl_base": dict(steps=40, guidance=7.0, model="sdxl", lora=None),
    "sd15_base": dict(steps=40, guidance=7.5, model="sd15", lora=None),
    "sd15_lora": dict(steps=40, guidance=7.5, model="sd15", lora="output/lora"),
}


def generate(variant_name, output_dir, device):
    cfg = VARIANTS[variant_name]
    pipe = load_pipeline(cfg["model"], device=device)
    if cfg["lora"]:
        pipe.load_lora_weights(cfg["lora"])

    out = Path(output_dir) / variant_name
    out.mkdir(parents=True, exist_ok=True)

    timings = []
    for i, prompt in enumerate(PROMPTS):
        start = time.time()
        image = pipe(
            prompt,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
            height=RESOLUTION,
            width=RESOLUTION,
        ).images[0]
        timings.append(time.time() - start)
        image.save(out / f"{i:02d}.png")

    (out / "timings.json").write_text(json.dumps(timings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS.keys())
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    generate(args.variant, args.output, args.device)
