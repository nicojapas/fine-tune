import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

SD15_MODEL = "runwayml/stable-diffusion-v1-5"
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
RESOLUTION = 512


def load_pipeline(model, device="cuda", dtype=torch.float16):
    if model == "sd15":
        pipe = StableDiffusionPipeline.from_pretrained(
            SD15_MODEL, dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
    elif model == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(SDXL_MODEL, dtype=dtype, variant="fp16")
    else:
        raise ValueError(f"unknown model: {model}")
    return pipe.to(device, dtype)
