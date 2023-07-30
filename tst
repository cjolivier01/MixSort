#!/bin/bash
# track on SportsMOT
# you can set different parameters for basketball, volleyball, and football in SportsMOT for better results
PYTHONPATH=$(pwd)/../yolox:$(pwd) \
  python \
    tools/track_mixsort.py \
    -expn yolox_m \
    -f exps/example/mot/yolox_x_sportsmot.py \
    -c pretrained/yolox_x_sports_train.pth \
    -b 1 -d 1 \
    --config track

