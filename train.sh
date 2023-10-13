#./p tools/train.py -f exps/example/mot/yolox_x_ch.py -d 8 -b 48 --fp16 -o
#./p tools/train.py -f exps/example/mot/yolox_x_ch.py -d 8 -b 24 --fp16
./p tools/train.py -f exps/example/mot/yolox_x_ch.py -d 8 -b 32 --fp16  -c pretrained/yolox_x.pth
