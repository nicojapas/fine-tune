import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from checks import passes
from prompts import PROMPTS
from score_clip import style_similarity

VARIANTS = ["sdxl_base", "sd15_base", "sd15_lora"]
LABELS = {
    "sdxl_base": "SDXL base (40-step)",
    "sd15_base": "SD1.5 base (40-step)",
    "sd15_lora": "SD1.5 + LoRA (40-step)",
}


def correctness_score(variant_dir):
    paths = sorted(Path(variant_dir).glob("*.png"))
    results = [passes(Image.open(p)) for p in paths]
    return sum(results) / len(results)


def mean_latency(variant_dir):
    timings = json.loads((Path(variant_dir) / "timings.json").read_text())
    return sum(timings) / len(timings)


def build_grid(output_dir, save_path):
    columns = [[Image.open(Path(output_dir) / v / f"{i:02d}.png") for i in range(len(PROMPTS))] for v in VARIANTS]
    w, h = columns[0][0].size
    grid = Image.new("RGB", (w * len(VARIANTS), h * len(PROMPTS)), "white")
    for col, images in enumerate(columns):
        for row, image in enumerate(images):
            grid.paste(image, (col * w, row * h))
    grid.save(save_path)


def build_scatter(metrics, save_path):
    fig, ax = plt.subplots()
    for variant, m in metrics.items():
        ax.scatter(m["latency"], m["correctness"])
        ax.annotate(LABELS[variant], (m["latency"], m["correctness"]))
    ax.set_xscale("log")
    ax.set_xlabel("latency per image (s)")
    ax.set_ylabel("correctness score")
    fig.savefig(save_path)


def main(output_dir, ref_dir, device):
    metrics = {}
    for variant in VARIANTS:
        variant_dir = Path(output_dir) / variant
        metrics[variant] = dict(
            correctness=correctness_score(variant_dir),
            clip=style_similarity(variant_dir, ref_dir, device),
            latency=mean_latency(variant_dir),
        )

    print(f"{'variant':<24}{'correctness':>12}{'clip':>10}{'latency':>10}")
    for variant, m in metrics.items():
        print(f"{variant:<24}{m['correctness']:>12.2f}{m['clip']:>10.3f}{m['latency']:>10.2f}")

    build_grid(output_dir, Path(output_dir) / "grid.png")
    build_scatter(metrics, Path(output_dir) / "scatter.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--ref_dir", default="data/sprites")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.output, args.ref_dir, args.device)
