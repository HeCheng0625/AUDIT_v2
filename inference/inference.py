from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from transformers import T5EncoderModel, T5TokenizerFast
import torch
import os
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
import torchaudio
import json

torch_device = torch.device("cuda:7")
check_point = "430000"
save_path = "/blob/v-yuancwang/AUDITDATA/AUDIT_INFER"
num_inference_steps = 100
guidance_scale= 7.5

model_path = "/blob/v-yuancwang/AudioEditingModel/Diffusion_SG/checkpoint-350000"
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained("/blob/v-yuancwang/AUDITPLUS/AUDIT_G_0/checkpoint-{}".format(check_point))
tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

os.makedirs(os.path.join(save_path, check_point+"_"+str(guidance_scale), "mel"), exist_ok=True)
save_path = os.path.join(save_path, check_point+"_"+str(guidance_scale), "mel")

def inference(mel_path, text, save_path, num_inference_steps=100, guidance_scale=7.5):
    prompt = [text]
    text_input = tokenizer(prompt, max_length=tokenizer.model_max_length, truncation=True, padding="do_not_pad", return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)

    scheduler.set_timesteps(num_inference_steps)

    latents = torch.randn((1, 4, 10, 78)).to(torch_device)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents_out = latents
    with torch.no_grad():
        res = vae.decode(latents_out).sample
    res = res.cpu().numpy()[0,0,:,:]
    
    np.save(os.path.join(save_path, mel_path), res)

with open("/home/v-yuancwang/AUDIT_v2/medata_infos/ac_test.json", "r") as f:
    test_infos = json.load(f)
for info in test_infos[:]:
    text = info["caption"]
    mel_path = info["mel"].split("/")[-1]
    inference(mel_path, text, save_path, num_inference_steps, guidance_scale)
