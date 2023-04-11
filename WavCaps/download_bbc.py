import os
from shutil import copy
import librosa
import numpy as np
import json
import wget
import requests
from tqdm import tqdm

bbc_json_file = "/home/v-yuancwang/DiffAudioImg/WavCaps/data/json_files/BBC_Sound_Effects/bbc_final.json"
with open(bbc_json_file, "r") as f:
    bbc_infos = json.load(f)
bbc_infos = bbc_infos['data']
print(len(bbc_infos))
print(bbc_infos[:5])

save_path = "/blob/v-yuancwang/WavCaps/BBC/wav_origin"

for info in tqdm(bbc_infos[:]):
    url = info["download_link"]
    id = info['id']
    if info['duration'] > 50:
        continue
    try:
        myfile = requests.get(url)
        open(os.path.join(save_path, id+".zip"), 'wb').write(myfile.content)
    except:
        continue