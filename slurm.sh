#!/bin/bash

#PARTITION_NAME=sw-a100
#PARTITION_NAME=sw-dpu
PARTITION_NAME=sw-mpu
GPUS_PER_HOST=8
BATCH_SIZE_PER_GPU=3
#GRES="--gres dpu:1"
#GRES="--gres gpu:8"
#NODE_COUNT="-N 35"
#NODE_COUNT="-N 1"
NODE_COUNT=23
START_EPOCH=51
TOTAL_BATCH_SIZE=$(( $GPUS_PER_HOST * $BATCH_SIZE_PER_GPU * $NODE_COUNT ))
echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"

#NODELIST="--nodelist=mojo-26u-r06u[03,05]"

NODELIST="--nodelist=mojo[011-018],mojo-26u-r06u[03,05,07,09,11,13,15,17,21,25,27,29,31,33,35,37]"
unset NODE_COUNT

RESUME="--resume"

#EXP="exps/example/mot/yolox_x_hockey_train.py"
#PRETRAINED_CHECKPOINT="pretrained/yolox_x_sports_train.pth"
#PRETRAINED_CHECKPOINT="./YOLOX_outputs/yolox_x_hockey_train/latest_ckpt.pth.tar"

EXP="exps/example/mot/yolox_x_ch.py"
#PRETRAINED_CHECKPOINT="pretrained/yolox_x.pth"
PRETRAINED_CHECKPOINT="YOLOX_outputs/yolox_x_ch/latest_ckpt.pth.tar"


srun --tasks-per-node 1 \
    ${NODE_COUNT} \
    ${NODELIST} \
    ${GRES} \
    --cpus-per-task=90 \
    -p ${PARTITION_NAME}  \
    --exclusive \
      ./p tools/train.py \
      -f "${EXP}" \
      -d ${GPUS_PER_HOST} \
      -b ${TOTAL_BATCH_SIZE} \
      --start_epoch=${START_EPOCH} \
      ${RESUME} \
      --fp16 \
      -c "${PRETRAINED_CHECKPOINT}"
