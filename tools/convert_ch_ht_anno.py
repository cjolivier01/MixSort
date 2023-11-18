# convert mot to trackingNet format
import os
from tqdm import tqdm


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

DATA_PATH = "datasets/hockeyTraining"
GT_INPUT_DATA_PATHS = [
    "datasets/hockeyTrackingDataset",
]
SPLITS = ["train", "test"]
OUT_PATH = os.path.join(DATA_PATH, "tracking_annos")

for split in SPLITS:
    out_path = os.path.join(OUT_PATH, split)
    mkdir(out_path)
    for data_path in GT_INPUT_DATA_PATHS:
        for seq in tqdm(os.listdir(os.path.join(data_path, split))):
            print(f"Processing sequence: {seq}")
            gt_path = os.path.join(data_path, split, seq, "gt.txt")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(data_path, split, seq, "gt", "gt.txt")
                if not os.path.exists(gt_path):
                    print(f"Path does not exist, skipping: {gt_path}")
                continue
            # read gt of all players
            with open(gt_path, "r") as f:
                gt = f.readlines()

            # key: id, value: gt of player(id)
            players = {}
            for line in gt:
                #print(line)
                line = line.split(",")
                frame, id = map(int, line[:2])
                x, y, w, h = map(int, line[2:6])
                if id not in players.keys():
                    players[id] = dict()
                this_player = players[id]
                #assert frame not in this_player
                if frame in this_player:
                    print(f"{seq}: player {id}, duplicate frame {frame}")
                this_player[frame] = (frame, x, y, w, h)
            # output, there is no need to sort using `frame`
            length = len(os.listdir(os.path.join(data_path, split, seq, "img1")))

            for id in players.keys():
                this_player = players[id]
                with open(os.path.join(out_path, f"{seq}-{id:0>3d}.txt"), "w") as f:
                    # start from frame 1
                    i = 0
                    for cur in range(1, length + 1):
                        try:
                            frame, x, y, w, h = this_player[i][cur]
                        except:
                            f.write("0,0,0,0\n")
                            continue
                            #i = cur
                            #break
                            pass
                        if frame != cur:
                            #f.write("0,0,0,0\n")
                            assert False
                        else:
                            f.write(f"{x},{y},{w},{h}\n")
                            i += 1
                    while i <= length:
                        f.write("0,0,0,0\n")
                        i += 1
