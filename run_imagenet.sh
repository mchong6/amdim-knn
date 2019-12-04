python -W ignore::UserWarning train.py \
  --ndf 192 \
  --n_rkhs 1536 \
  --batch_size 480 \
  --tclip 20.0 \
  --n_depth 8 \
  --dataset IN128 \
  --amp \
  --output_dir ./imagenet_runs

