#!/bin/bash

#PARTITION_NAME=sw-a100
#PARTITION_NAME=sw-dpu
PARTITION_NAME=sw-all
GPUS_PER_HOST=8
BATCH_SIZE_PER_GPU=2
#GRES="--gres dpu:1"
#GRES="--gres gpu:8"
#NODE_COUNT="-N 35"
#NODE_COUNT="-N 1"
#START_EPOCH="--start_epoch=0"

#NODELIST="--nodelist=mojo-26u-r06u[03,05]"

#NODE_COUNT=23
#NODELIST="--nodelist=mojo[011-018],mojo-26u-r06u[03,05,07,09,11,13,15,17,21,25,27,29,31,33,35,37]"

#NODELIST="${NODELIST},mojo-ep2-r02u[01,03,07,11,13,21,25,27,45,47]"
#NODE_COUNT=33

NODELIST="--nodelist=mojo-26l-r202u[03,05,07,11,13,15,17,19,21,25,27,45,47]"
NODE_COUNT=13

#NODELIST="--nodelist=mojo-26l-r202u01"
#NODE_COUNT=1

TOTAL_BATCH_SIZE=$(( $GPUS_PER_HOST * $BATCH_SIZE_PER_GPU * $NODE_COUNT ))
echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"

unset NODE_COUNT

RESUME="--resume"

#EXP="exps/example/mot/yolox_x_hockey_train.py"
#PRETRAINED_CHECKPOINT="pretrained/yolox_x_sports_train.pth"
#PRETRAINED_CHECKPOINT="./YOLOX_outputs/yolox_x_hockey_train/latest_ckpt.pth.tar"

#EXP="exps/example/mot/yolox_m_ch.py"
#EXP="exps/example/mot/yolox_x_ch.py"
#EXP="exps/example/mot/yolox_x_ch.py"
#EXP="exps/example/mot/yolox_x_hockey_train2.py"
EXP="exps/example/mot/yolox_x_ch_ht.py"
#PRETRAINED_CHECKPOINT="pretrained/my_ch.pth.tar"
#PRETRAINED_CHECKPOINT="YOLOX_outputs/yolox_x_ch/latest_ckpt.pth.tar"

PYTHONPATH="$(pwd)/../../src:$PYTHONPATH" \
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
      -o \
      ${START_EPOCH} \
      ${RESUME} \
      --fp16
