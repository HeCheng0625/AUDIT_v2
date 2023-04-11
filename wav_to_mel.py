from scipy.io.wavfile import read
import torchaudio
import torch
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os
from tqdm import tqdm
import sys

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

SUBSET = "vggsound"
wav_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/{}/wav".format(SUBSET)
mel_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/{}/mel".format(SUBSET)
wav_list = os.listdir(wav_path)
mel_set = set(os.listdir(mel_path))

wav_list.sort()
for wav_file in tqdm(wav_list):
    if wav_file.replace(".wav", ".npy") in mel_set:
        continue
    wav_file_path = os.path.join(wav_path, wav_file)    
    wav, sr = librosa.load(wav_file_path, sr=16000)
    # print(len(wav), sr)
    # print(wav.shape)
    if len(wav) < 16000:
        continue
    wav = torch.FloatTensor(wav)
    # print(wav.unsqueeze(0).shape)
    x = mel_spectrogram(wav.unsqueeze(0), n_fft=1024, num_mels=80, sampling_rate=16000,
                        hop_size=256, win_size=1024, fmin=0, fmax=8000)
    # print(x.shape, x.max(), x.min())
    # print(x)
    spec = x.cpu().numpy()[0]
    # print(spec.shape)
    np.save(os.path.join(mel_path, wav_file.replace(".wav", ".npy")), spec)