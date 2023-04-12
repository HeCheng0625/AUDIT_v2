export MODEL_NAME=""
export TRAIN_DIR=""

accelerate launch train_vae_gan.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=6 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=500000 \
  --checkpointing_steps=5000 \
  --learning_rate=8e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --output_dir="/blob/v-yuancwang/AUDITPLUS/VAE_GAN"