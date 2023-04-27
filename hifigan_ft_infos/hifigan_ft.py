from diffusers import AutoencoderKL
import torch
import os
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
import torchaudio
import json

vae = AutoencoderKL.from_pretrained("/blob/v-yuancwang/AudioEditingModel/VAE_GAN/checkpoint-40000/vae")
vae.requires_grad_(False)
torch_device = torch.device("cuda:5")
vae.to(torch_device)

train_json = "/home/v-yuancwang/AUDIT_v2/medata_infos/vggsound.json"
with open(train_json, "r") as f:
    train_list = json.load(f)
print(train_list[:2])

vae_lists = []

for info in tqdm(train_list[:]):
    mel_path = info["mel"]
    wav_path = mel_path.replace("/mel/", "/wav/").replace(".npy", ".wav")
    wav, sr = librosa.load(wav_path, sr=16000)
    if len(wav) < 159500:
        continue
    mel = np.load(mel_path)

    test_mel_tensor = np.expand_dims(np.expand_dims(mel[:624], 0), 0)
    test_mel_tensor = torch.Tensor(test_mel_tensor).to(torch_device)
    with torch.no_grad():
        posterior = vae.encode(test_mel_tensor).latent_dist
        z = posterior.sample()
        vae_output = vae.decode(z).sample
    vae_res = vae_output[0][0].cpu().numpy()

    np.save(mel_path.replace("/mel/", "/vae_mel/"), vae_res)
    vae_lists.append(info)

vae_json = train_json.replace("/medata_infos/","/hifigan_ft_infos/")
with open(vae_json, "w") as f:
    json.dump(vae_lists, f)