#!/bin/bash
OMP_NUM_THREADS=16 \
  ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey
