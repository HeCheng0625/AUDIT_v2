import os

infer_paths = os.listdir("/blob/v-yuancwang/AUDITDATA/AUDIT_CD_INFER") 
for pth in infer_paths:
    mel_path = os.path.join("/blob/v-yuancwang/AUDITDATA/AUDIT_CD_INFER", pth, "mel")
    wav_path = os.path.join("/blob/v-yuancwang/AUDITDATA/AUDIT_CD_INFER", pth, "wav")
    os.makedirs(wav_path, exist_ok=True)
    os.system('python /home/v-yuancwang/AUDIT_v2/hifi-gan/inference_e2e.py\
            --input_mels_dir="{}" \
            --output_dir="{}" \
            --checkpoint_file="/blob/v-yuancwang/hifigan_cp/g_01250000"'.format(mel_path, wav_path))