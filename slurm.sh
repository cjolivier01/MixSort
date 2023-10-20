#!/bin/bash

PARTITION_NAME=sw-a100
#PARTITION_NAME=sw-dpu
GPUS_PER_HOST=8
BATCH_SIZE_PER_GPU=8
NODE_COUNT=1
#GRES=dpu:1
GRES=gpu:8
#NODE_COUNT=25
START_EPOCH=8
TOTAL_BATCH_SIZE=$(( $GPUS_PER_HOST * $BATCH_SIZE_PER_GPU * $NODE_COUNT ))
echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"

#PRETRAINED_CHECKPOINT="pretrained/yolox_x_sports_train.pth"
PRETRAINED_CHECKPOINT="./YOLOX_outputs/yolox_x_hockey_train/latest_ckpt.pth.tar"

srun --tasks-per-node 1 \
    -N ${NODE_COUNT} \
    --gres ${GRES} \
    --cpus-per-task=90 \
    -p ${PARTITION_NAME}  \
    --exclusive \
      ./p tools/train.py \
      -f exps/example/mot/yolox_x_hockey_train.py \
      -d ${GPUS_PER_HOST} \
      -b ${TOTAL_BATCH_SIZE} \
      --start_epoch=${START_EPOCH} \
      --resume \
      --fp16 \
      -c "${PRETRAINED_CHECKPOINT}"
