#!/bin/bash

# OMP_NUM_THREADS=16 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey
export WORLD_SIZE=8
for i in {0..7}; do
  echo $i
   WORLD_SIZE=$WORLD_SIZE \
   MASTER_ADDR="127.0.0.1" \
   MASTER_PORT=34595 \
   RANK=$i \
   ./p MixViT/lib/train/run_training.py --local_rank=$i --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey > mf-train-$i.log 2>&1 &
done

# OMP_NUM_THREADS=16 \
#   WORLD_SIZE=8 \
#   RANK=0 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey &

# OMP_NUM_THREADS=16 \
#   WORLD_SIZE=8 \
#   RANK=1 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey &

# OMP_NUM_THREADS=16 \
#   WORLD_SIZE=8 \
#   RANK=2 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey &

# OMP_NUM_THREADS=16 \
#   WORLD_SIZE=8 \
#   RANK=3 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey &

# OMP_NUM_THREADS=16 \
#   WORLD_SIZE=8 \
#   RANK=4 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey &

# OMP_NUM_THREADS=16 \
#   WORLD_SIZE=8 \
#   RANK=5 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey &

# OMP_NUM_THREADS=16 \
#   WORLD_SIZE=8 \
#   RANK=6 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey &

# OMP_NUM_THREADS=16 \
#   WORLD_SIZE=8 \
#   RANK=7 \
#   ./p MixViT/lib/train/run_training.py --script mixformer_deit_hockey --config baseline --save_dir=./exp/mixformer_deit_hockey
