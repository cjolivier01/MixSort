#./p tools/train.py -f exps/example/mot/yolox_x_ch.py -d 8 -b 48 --fp16 -o
#./p tools/train.py -f exps/example/mot/yolox_x_ch.py -d 8 -b 24 --fp16
#./p tools/train.py -f exps/example/mot/yolox_x_ch.py -d 8 -b 16 --fp16 -o -c pretrained/yolox_x.pth
#./p tools/train.py -f exps/example/mot/yolox_x_hockey.py -d 8 -b 16 --fp16 -c pretrained/yolox_x_sports_train.pth
OMP_NUM_THREADS=16 \
  ./p tools/train.py --resume --start_epoch=12 -f exps/example/mot/yolox_x_hockey_train.py -d 8 -b 48 --fp16 -c YOLOX_outputs/yolox_x_hockey_train/latest_ckpt.pth.tar $@

