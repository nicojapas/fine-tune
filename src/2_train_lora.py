import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image

from models import load_pipeline
from prompts import TRIGGER


def image_to_tensor(image, dtype):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dtype)


def train(data_dir, output_dir, epochs, lr, device):
    dtype = torch.float16
    pipe = load_pipeline("sd15", device=device, dtype=dtype)
    pipe.unet.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.to(torch.float32)
    optimizer = torch.optim.AdamW(params, lr=lr)
    pipe.vae.to(torch.float32)

    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(TRIGGER, device, 1, False)

    num_train_timesteps = pipe.scheduler.config.num_train_timesteps
    paths = sorted(Path(data_dir).glob("*.png"))

    for epoch in range(epochs):
        for path in paths:
            pixel_values = image_to_tensor(Image.open(path), dtype).to(device)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values.to(pipe.vae.dtype)).latent_dist.sample() * pipe.vae.config.scaling_factor
            latents = latents.to(dtype)

            timestep = torch.randint(0, num_train_timesteps, (1,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)

            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype):
                noise_pred = pipe.unet(noisy_latents, timestep, encoder_hidden_states=prompt_embeds).sample

            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"epoch {epoch} loss {loss.item():.4f}")

    lora_state_dict = get_peft_model_state_dict(pipe.unet)
    pipe.save_lora_weights(output_dir, unet_lora_layers=lora_state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--output", default="output/lora")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.lr, args.device)
