#!/bin/bash
OMP_NUM_THREADS=16 \
  MASTER_PORT="29501" \
  MASTER_ADDRESS="192.168.221.61" \
  ./p tools/train.py \
    --local_rank=0 \
    --start_epoch=4 \
    --num_machines=4 \
    --machine_rank=2 \
    -f exps/example/mot/yolox_x_hockey.py -d 8 -b 64 --fp16 -c YOLOX_outputs/yolox_x_hockey/latest_ckpt.pth.tar
