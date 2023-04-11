import os
import random
wavs = os.listdir("/blob/v-yuancwang/audio_editing_data/audiocaps/wav")
random.shuffle(wavs)
with open("/home/v-yuancwang/AudioEditing/hifi-gan/audiocaps_train.txt", 'w') as f:
    for wav in wavs[:-10]:
        f.write(wav.replace(".wav", "").replace("\n", "") + "\n")
with open("/home/v-yuancwang/AudioEditing/hifi-gan/audiocaps_val.txt", 'w') as f:
    for wav in wavs[-10:]:
        f.write(wav.replace(".wav", "").replace("\n", "") + "\n")