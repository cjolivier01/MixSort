OMP_NUM_THREADS=16 \
  ./p tools/train.py -f exps/example/mot/yolox_x_ch_hockey_train.py -d 1 -b 3 --fp16 $@
