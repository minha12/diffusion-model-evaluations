python scripts/preparation/convert_controlnet.py \
  --checkpoint_path /home/ubuntu/models/trained-diffusion/sd21_controlnet/controlnet/epoch=48-step=38856.ckpt \
  --original_config_file /home/ubuntu/models/trained-diffusion/sd21_controlnet/controlnet/config.yaml \
  --dump_path /home/ubuntu/models/trained-diffusion/sd21_controlnet/controlnet \
  --image_size 512 \
  --num_in_channels 4 \
  --cross_attention_dim 1024 \
  --use_linear_projection True \
  --to_safetensors