#!/bin/bash
#RUN_PARTITION_SERVERS="mojo017,mojo-26u-r06u[03,05,07,09]"

#srun --tasks-per-node 8 -N 4 -p sw-mpu --cpus-per-task 12 \

#  srun --tasks-per-node 8 --nodelist $RUN_PARTITION_SERVERS -p sw-mpu --cpus-per-task 12 \

#--nodelist mojo-ep2-r02u47 \

srun --tasks-per-node 1 \
  -N 2 \
  --cpus-per-task=90 \
  -p sw-mpu  \
  --exclusive \
    ./p tools/train.py -f exps/example/mot/yolox_x_hockey.py -d 8 -b 32 --fp16 -c ./YOLOX_outputs/yolox_x_hockey/latest_ckpt.pth.tar
