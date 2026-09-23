import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_clip(device):
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def embed_images(paths, model, processor, device):
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(vision_outputs.pooler_output)
    return features / features.norm(dim=-1, keepdim=True)


def style_similarity(gen_dir, ref_dir, device="cuda"):
    model, processor = load_clip(device)
    ref_paths = sorted(Path(ref_dir).glob("*.png"))
    gen_paths = sorted(Path(gen_dir).glob("*.png"))
    ref_embeds = embed_images(ref_paths, model, processor, device)
    gen_embeds = embed_images(gen_paths, model, processor, device)
    sims = gen_embeds @ ref_embeds.T
    return sims.max(dim=1).values.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_dir")
    parser.add_argument("--ref_dir", default="data/sprites")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(style_similarity(args.gen_dir, args.ref_dir, args.device))
