/opt/conda/envs/control/bin/python /home/v-yuancwang/AUDIT_v2/hifi-gan/train.py \
--input_wavs_dir="" \
--input_training_file="" \
--input_validation_file="" \
--checkpoint_path="/blob/v-yuancwang/hifigan_cp" \
--config="/home/v-yuancwang/AUDIT_v2/hifi-gan/config_ours.json" \
--training_epochs=800 \
--stdout_interval=5 \
--checkpoint_interval=5000 \
--summary_interval=100 \
--validation_interval=1000 \
--fine_tuning=True \
