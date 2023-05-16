from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler, DDPMScheduler
from transformers import T5EncoderModel, T5TokenizerFast
import torch
import os
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
import torchaudio
from IPython.display import Audio
import matplotlib.pyplot as plt
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch_device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default="1",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    return args

args = parse_args()

torch_device = torch.device(args.torch_device)
num_inference_steps = args.num_inference_steps
guidance_scale = args.guidance_scale
checkpoint = args.checkpoint

model_path = "/blob/v-yuancwang/AudioEditingModel/Diffusion_SG/checkpoint-350000"

unet_path = "/blob/v-yuancwang/AUDITPLUS/AUDIT_CD_100/checkpoint-{}".format(str(checkpoint))
save_path = "/blob/v-yuancwang/AUDITDATA/AUDIT_CD_INFER"
save_path = os.path.join(save_path, "infer_step_{}_gs_{}_cp_{}".format(str(num_inference_steps), str(guidance_scale), str(checkpoint)), "wav")
os.makedirs(save_path, exist_ok=True)

vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(unet_path)
tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

with open("/home/v-yuancwang/AUDIT_v2/medata_infos/ac_test.json", "r") as f:
    infos = json.load(f)
for info in infos[:]:
    mel, caption = info["mel"], info["caption"]
    mel_path = mel.split("/")[-1]
    print(mel_path)
    text = caption

    prompt = [text]
    text_input = tokenizer(prompt, max_length=tokenizer.model_max_length, truncation=True, padding="do_not_pad", return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
    print(text_embeddings.shape)
    print(uncond_embeddings.shape)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    noisy_latents = torch.randn((1, 4, 10, 78)).to(torch_device)
    nosiy_latents_input = torch.cat([noisy_latents] * 2)

    with torch.no_grad():
        timesteps = torch.tensor(990).to(torch_device)
        c_skip = 0.25 / (0.25 + (timesteps / 1000 - 0.0) ** 2)
        c_out = 0.5 * (timesteps / 1000 - 0.0) / ((timesteps / 1000) ** 2 + 0.25) ** 0.5
        pred_target = c_skip.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * nosiy_latents_input + c_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * unet(nosiy_latents_input, timesteps, text_embeddings).sample
        
        pred_target_uncond, pred_target_cond = pred_target.chunk(2)
        pred_target = pred_target_uncond + guidance_scale * (pred_target_cond - pred_target_uncond)

    if num_inference_steps == 2:
        z = torch.randn((1, 4, 10, 78)).to(torch_device)
        noisy_latents = scheduler.add_noise(pred_target, z, torch.tensor(500).to(torch_device))

        nosiy_latents_input = torch.cat([noisy_latents] * 2)

        with torch.no_grad():
            timesteps = torch.tensor(990).to(torch_device)
            c_skip = 0.25 / (0.25 + (timesteps / 1000 - 0.0) ** 2)
            c_out = 0.5 * (timesteps / 1000 - 0.0) / ((timesteps / 1000) ** 2 + 0.25) ** 0.5
            pred_target = c_skip.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * nosiy_latents_input + c_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * unet(nosiy_latents_input, timesteps, text_embeddings).sample
            
            pred_target_uncond, pred_target_cond = pred_target.chunk(2)
            pred_target = pred_target_uncond + guidance_scale * (pred_target_cond - pred_target_uncond)

    latents_out = pred_target
    with torch.no_grad():
        res = vae.decode(latents_out).sample
    res = res.cpu().numpy()[0,0,:,:]
    np.save(os.path.join(save_path, mel_path.replace(".npy", "")), res)