#!/bin/sh

python main.py \
  --env_name=puzzle-4x6-play-oraclerep-v0 \
  --dataset_dir=datasets/puzzle-4x6-play-1b-v0 \
  --agent=agents/sharsa.py \
  --wandb_mode online \
  --num_datasets 1000 \