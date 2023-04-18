from scipy.io.wavfile import read, write
import torchaudio
import torch
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import os
import soundfile as sf
import matplotlib.pyplot as plt

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
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

# wav_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/ac_train/wav"
# sr_wav_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/ac_train/sr/wav"
# sr_mel_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/ac_train/sr/mel"

def gen_sr(wav_path, sr_wav_path, sr_mel_path):
    wav_files = os.listdir(wav_path)
    print(len(wav_files))

    sr_wav_set = set(os.listdir(sr_wav_path))
    print(len(sr_wav_set))


    for file_name in tqdm(wav_files[:]):
        if file_name in sr_wav_set:
            continue
        try:
            mel = np.load(os.path.join(sr_mel_path, file_name.replace(".wav", ".npy")))
            print(mel.shape)
            wav, sr = librosa.load(os.path.join(wav_path, file_name), sr=8000)
            wav = np.clip(wav, -1, 1)
            wav = wav * MAX_WAV_VALUE
            wav = wav.astype('int16')
            write(os.path.join(sr_wav_path, file_name), 8000, wav)

            wav, sr = librosa.load(os.path.join(sr_wav_path, file_name), sr=16000)
            wav = np.clip(wav, -1, 1)
            x = torch.FloatTensor(wav)
            # print(len(x))
            x = mel_spectrogram(x.unsqueeze(0), n_fft=1024, num_mels=80, sampling_rate=16000,
                            hop_size=256, win_size=1024, fmin=0, fmax=8000)
            # print(x.shape)
            spec = x.cpu().numpy()[0]
            if spec.shape[1] < 624:
                spec = np.pad(spec, ((0, 0), (0, 624 - spec.shape[1])), 'wrap')
            # print(spec.shape)
            wav = wav * MAX_WAV_VALUE
            wav = wav.astype('int16')
            write(os.path.join(sr_wav_path, file_name), 16000, wav)
            np.save(os.path.join(sr_mel_path, file_name.replace(".wav", ".npy")), spec) 

        except:
            continue

gen_sr()
gen_sr()
gen_sr()