import os
import numpy as np
import json
import cv2
import argparse


# Use the same script for MOT16
DATA_PATH = "datasets/norcal"
OUT_PATH = os.path.join(DATA_PATH, "annotations")
# SPLITS = ['train_half', 'val-_half', 'train', 'test']  # --> split training data to train_half and val_half.
SPLITS = ["train", "test"]  # --> split training data to train_half and val_half.
HALF_VIDEO = False
CREATE_SPLITTED_ANN = False
CREATE_SPLITTED_DET = False


def _is_image(file: str):
    upper = file.lower()
    return upper.endswith(".png")

def _tf_str(val: bool):
  if val:
    return "true"
  return "false"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert Norcal to COCO")
    parser.add_argument("--single-sequence", type=str, default=None)
    parser.add_argument(
        "--input-image-prefix",
        type=str,
        default="",
        help="Prefix input file names",
    )
    parser.add_argument(
        "--output-image-prefix",
        type=str,
        default="",
        help="Prefix file names in annotations (i.e. CVAT prepends frame_ to exploded video images)",
    )
    parser.add_argument(
        "--image-start-number",
        type=int,
        default=1,
        help="Start number of the frame image files (default = 1)",
    )
    parser.add_argument(
        "--ignore-track-ids",
        default="",
        type=str,
        help="Comma-delimited list of track id's to set as 'ignore'",
    )
    parser.add_argument(
        "--no-ignore",
        default=False,
        action="store_true",
        help="Don't treat any annotations as ignored",
    )
    parser.add_argument(
        "--cvat",
        default=False,
        action="store_true",
        help="CVAT-friendly output",
    )
    args = parser.parse_args()

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    seen_categories = set()

    ignore_track_ids = set()
    if args.ignore_track_ids:
      ignore_list = args.ignore_track_ids.split(',')
      ignore_track_ids = set([int(i) for i in ignore_list])

    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, "test")
        else:
            data_path = os.path.join(DATA_PATH, "train")
        out_path = os.path.join(OUT_PATH, "{}.json".format(split))
        out = {
            "images": [],
            "annotations": [],
            "videos": [],
            "categories": [{"id": 1, "name": "pedestrian"}],
        }
        seqs = sorted(os.listdir(data_path))
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        for seq in sorted(seqs):
            if ".DS_Store" in seq:
                continue
            video_cnt += 1  # video sequence number.
            out["videos"].append({"id": video_cnt, "file_name": seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, "img1")
            ann_path = os.path.join(seq_path, "gt", "gt.txt")
            # if not os.path.exists(img_path):
            #     print(f"Path does not exist (skipping): {img_path}")
            #     continue
            if not os.path.exists(ann_path):
                print(f"Annotation path does not exist (skipping): {ann_path}")
                continue
            images = os.listdir(img_path)
            num_images = len(
                [image for image in images if _is_image(image)]
            )  # half and half

            if HALF_VIDEO and ("half" in split):
                image_range = (
                    [0, num_images // 2]
                    if "train" in split
                    else [num_images // 2 + 1, num_images - 1]
                )
            else:
                image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                input_file_name = os.path.join(
                    seq, "img1", "{}{:06d}.png".format(args.input_image_prefix, i + args.image_start_number)
                )
                output_file_name = ""
                if not args.single_sequence:
                    output_file_name = os.path.join(seq, "img1")
                else:
                    output_file_name = "img1"
                output_file_name = os.path.join(
                    output_file_name,
                    "{}{:06d}.png".format(args.output_image_prefix, i + args.image_start_number),
                )
                img = cv2.imread(os.path.join(data_path, input_file_name))
                height, width = img.shape[:2]
                image_info = {
                    "file_name": output_file_name,  # image name.
                    "id": image_cnt + i + 1,  # image number in the entire training set.
                    "frame_id": i
                    + 1
                    - image_range[
                        0
                    ],  # image number in the video sequence, starting from 1.
                    "prev_image_id": image_cnt + i
                    if i > 0
                    else -1,  # image number in the entire training set.
                    "next_image_id": image_cnt + i + 2 if i < num_images - 1 else -1,
                    "video_id": video_cnt,
                    "height": height,
                    "width": width,
                }
                out["images"].append(image_info)
            print("{}: {} images".format(seq, num_images))
            if split != "test" or True:
                det_path = os.path.join(seq_path, "det/det.txt")
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
                if CREATE_SPLITTED_ANN and ("half" in split):
                    anns_out = np.array(
                        [
                            anns[i]
                            for i in range(anns.shape[0])
                            if int(anns[i][0]) - 1 >= image_range[0]
                            and int(anns[i][0]) - 1 <= image_range[1]
                        ],
                        np.float32,
                    )
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, "gt/gt_{}.txt".format(split))
                    fout = open(gt_out, "w")
                    for o in anns_out:
                        fout.write(
                            "{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n".format(
                                int(o[0]),
                                int(o[1]),
                                int(o[2]),
                                int(o[3]),
                                int(o[4]),
                                int(o[5]),
                                int(o[6]),
                                int(o[7]),
                                o[8],
                            )
                        )
                    fout.close()
                if CREATE_SPLITTED_DET and ("half" in split):
                    dets = np.loadtxt(det_path, dtype=np.float32, delimiter=",")
                    dets_out = np.array(
                        [
                            dets[i]
                            for i in range(dets.shape[0])
                            if int(dets[i][0]) - 1 >= image_range[0]
                            and int(dets[i][0]) - 1 <= image_range[1]
                        ],
                        np.float32,
                    )
                    dets_out[:, 0] -= image_range[0]
                    det_out = os.path.join(seq_path, "det/det_{}.txt".format(split))
                    dout = open(det_out, "w")
                    for o in dets_out:
                        dout.write(
                            "{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n".format(
                                int(o[0]),
                                int(o[1]),
                                float(o[2]),
                                float(o[3]),
                                float(o[4]),
                                float(o[5]),
                                float(o[6]),
                            )
                        )
                    dout.close()

                print("{} ann images".format(int(anns[:, 0].max())))
                ignore_count = 0

                # Stats
                all_track_ids = set()
                non_person = 0
                ignored_persons = 0
                non_ignored_person = 0

                for i in range(anns.shape[0]):
                    this_ann = anns[i]
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    all_track_ids.add(track_id)
                    # if track_id:
                    #   print(f"track_id={track_id}")
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    ignore = 0
                    # if not (float(anns[i][8]) >= 0.25):  # visibility.
                    #     continue
                    if not args.no_ignore and not (
                        int(anns[i][6]) == 1
                    ):  # whether ignore.
                        ignore_count += 1
                        continue
                    if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                        non_person += 1
                        continue
                    if int(anns[i][7]) in [2, 7, 8, 12]:  # Ignored person
                        category_id = -1
                        ignore = 1
                        ignored_persons += 1
                    else:
                        category_id = 1  # pedestrian(non-static)
                        non_ignored_person += 1

                    if category_id not in seen_categories:
                        print(f"Category: {category_id}")
                        seen_categories.add(category_id)
                    if not args.cvat:
                        ann = {
                            "id": ann_cnt,
                            "category_id": category_id,
                            "image_id": image_cnt + frame_id,
                            "track_id": track_id,
                            "bbox": anns[i][2:6].tolist(),
                            "conf": float(anns[i][6]),
                            "iscrowd": 0,
                            "area": float(anns[i][4] * anns[i][5]),
                            "segmentation": [],
                            "ignore": ignore,
                        }
                    else:
                        ann = {
                            "id": ann_cnt,
                            "image_id": image_cnt + frame_id,
                            "category_id": category_id,
                            "segmentation": [],
                            "area": float(anns[i][4] * anns[i][5]),
                            "bbox": anns[i][2:6].tolist(),
                            "iscrowd": 0,
                            "conf": float(anns[i][6]),
                            "attributes": {
                                "ignore": _tf_str(ignore),
                                "occluded": _tf_str(False),
                                "rotation": 0.0,
                                "track_id": track_id,
                                "keyframe": _tf_str(True),
                            },
                        }
                    out["annotations"].append(ann)
            image_cnt += num_images
            print(
                f"{ignore_count} annotations ignored, "
                f"{non_ignored_person} unignored pedestrians, "
                f"{non_person} non-persons, "
                f"{ignored_persons} ignored people, "
                f"{len(all_track_ids)} unique track ids"
            )
            print(tid_curr, tid_last)
        print(
            "loaded {} for {} images and {} samples".format(
                split, len(out["images"]), len(out["annotations"])
            )
        )
        json.dump(out, open(out_path, "w"))
