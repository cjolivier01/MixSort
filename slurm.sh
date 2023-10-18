#!/bin/bash

GPUS_PER_HOST=4
BATCH_SIZE_PER_GPU=1
HOST_COUNT=2

TOTAL_BATCH_SIZE=$(( $GPUS_PER_HOST * $BATCH_SIZE_PER_GPU * $HOST_COUNT ))
echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"

srun --tasks-per-node 1 \
    -N ${HOST_COUNT} \
    --cpus-per-task=90 \
    -p sw-gpu  \
    --exclusive \
      ./p tools/train.py \
      -f exps/example/mot/yolox_x_hockey.py \
      -d ${GPUS_PER_HOST} \
      -b ${TOTAL_BATCH_SIZE} \
      --fp16 \
      -c ./YOLOX_outputs/yolox_x_hockey/latest_ckpt.pth.tar
