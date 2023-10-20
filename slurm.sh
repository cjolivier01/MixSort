#!/bin/bash

GPUS_PER_HOST=4
BATCH_SIZE_PER_GPU=3
#NODE_COUNT=14
NODE_COUNT=25
START_EPOCH=7
TOTAL_BATCH_SIZE=$(( $GPUS_PER_HOST * $BATCH_SIZE_PER_GPU * $NODE_COUNT ))
echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"

#PRETRAINED_CHECKPOINT="pretrained/yolox_x_sports_train.pth"
PRETRAINED_CHECKPOINT="./YOLOX_outputs/yolox_x_hockey_train/latest_ckpt.pth.tar"

srun --tasks-per-node 1 \
    -N ${NODE_COUNT} \
    --gres dpu:1 \
    --cpus-per-task=90 \
    -p sw-dpu  \
    --exclusive \
      ./p tools/train.py \
      -f exps/example/mot/yolox_x_hockey_train.py \
      -d ${GPUS_PER_HOST} \
      -b ${TOTAL_BATCH_SIZE} \
      --start_epoch=${START_EPOCH} \
      --resume \
      --fp16 \
      -c "${PRETRAINED_CHECKPOINT}"
