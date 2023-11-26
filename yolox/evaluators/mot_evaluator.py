from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
)

# from yolox.tracker.byte_tracker import BYTETracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker

from hmlib.tracker.multitracker import JDETracker

import contextlib
import io
import os
import itertools
import json
import tempfile
import time

# import pt_autograph
# import pt_autograph.flow.runner as runner

from hmlib.tracking_utils.timer import Timer


def write_results(filename, results):
    save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    s=round(score, 2),
                )
                f.write(line)
    logger.info("save results to {}".format(filename))


def write_results_no_score(filename, results):
    save_format = "{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                )
                f.write(line)
    logger.info("save results to {}".format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        args,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
        online_callback: callable = None,
        postprocessor=None,
        device: str = None,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.device = device
        if self.device is None:
            self.device = f"cuda:{args.local_rank}"
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args
        self.online_callback = online_callback
        self.timer = None
        self.timer_counter = 0
        self.track_timer = Timer()
        self.track_timer_counter = 0
        self.preproc_timer = Timer()
        self.preproc_timer_counter = 0
        self.postprocessor = postprocessor

    def filter_outputs(self, outputs: torch.Tensor, output_results):
        if self.postprocessor is not None:
            return self.postprocessor.filter_outputs(outputs, output_results)
        return outputs, output_results

    def evaluate_byte(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        if self.args.iou_only:
            from yolox.byte_tracker.byte_iou_tracker import BYTETracker
        else:
            from yolox.byte_tracker.byte_tracker import BYTETracker
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            # for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2][0]
                video_id = info_imgs[3][0].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[0]
                if video_name == "MOT17-05-FRCNN" or video_name == "MOT17-06-FRCNN":
                    self.args.track_buffer = 14
                elif video_name == "MOT17-13-FRCNN" or video_name == "MOT17-14-FRCNN":
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == "MOT17-01-FRCNN":
                    self.args.track_thresh = 0.65
                elif video_name == "MOT17-06-FRCNN":
                    self.args.track_thresh = 0.65
                elif video_name == "MOT17-12-FRCNN":
                    self.args.track_thresh = 0.7
                elif video_name == "MOT17-14-FRCNN":
                    self.args.track_thresh = 0.67
                elif video_name in ["MOT20-06", "MOT20-08"]:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(
                            result_folder, "{}.txt".format(video_names[video_id - 1])
                        )
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                if self.online_callback is not None:
                    _, online_tlwhs = self.online_callback(
                        frame_id=frame_id,
                        online_tlwhs=online_tlwhs,
                        online_ids=online_ids,
                        online_scores=online_scores,
                        info_imgs=info_imgs,
                        img=imgs,
                        original_img=origin_imgs,
                    )

                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(
                    result_folder, "{}.txt".format(video_names[video_id])
                )
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    # def evaluate_mixsort(
    #     self,
    #     model,
    #     distributed=False,
    #     half=False,
    #     trt_file=None,
    #     decoder=None,
    #     test_size=None,
    #     result_folder=None,
    # ):
    #     """
    #     COCO average precision (AP) Evaluation. Iterate inference on the test dataset
    #     and the results are evaluated by COCO API.

    #     NOTE: This function will change training mode to False, please save states if needed.

    #     Args:
    #         model : model to evaluate.

    #     Returns:
    #         ap50_95 (float) : COCO AP of IoU=50:95
    #         ap50 (float) : COCO AP of IoU=50
    #         summary (sr): summary info of evaluation.
    #     """
    #     if self.args.iou_only:
    #         from yolox.mixsort_tracker.mixsort_iou_tracker import MIXTracker
    #     else:
    #         from yolox.mixsort_tracker.mixsort_tracker import MIXTracker

    #     # like ByteTrack, we use different setting for different videos
    #     setting = {
    #         "MOT17-01-FRCNN": {"track_buffer": 27, "track_thresh": 0.6275},
    #         "MOT17-03-FRCNN": {"track_buffer": 31, "track_thresh": 0.5722},
    #         "MOT17-06-FRCNN": {"track_buffer": 16, "track_thresh": 0.5446},
    #         "MOT17-07-FRCNN": {"track_buffer": 24, "track_thresh": 0.5939},
    #         "MOT17-08-FRCNN": {"track_buffer": 24, "track_thresh": 0.7449},
    #         "MOT17-12-FRCNN": {"track_buffer": 29, "track_thresh": 0.7036},
    #         "MOT17-14-FRCNN": {"track_buffer": 28, "track_thresh": 0.5436},
    #     }

    #     def set_args(args, video):
    #         if video not in setting.keys():
    #             return args
    #         for k, v in setting[video].items():
    #             args.__setattr__(k, v)
    #         return args

    #     # TODO half to amp_test
    #     tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    #     model = model.eval()
    #     if half:
    #         model = model.half()
    #     ids = []
    #     data_list = []
    #     results = []
    #     video_names = defaultdict()
    #     progress_bar = tqdm if is_main_process() and not self.online_callback else iter

    #     inference_time = 0
    #     track_time = 0
    #     n_samples = len(self.dataloader) - 1

    #     use_autograph = False

    #     if trt_file is not None:
    #         from torch2trt import TRTModule

    #         model_trt = TRTModule()
    #         model_trt.load_state_dict(torch.load(trt_file))

    #         x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
    #         model(x)
    #         model = model_trt

    #     tracker = MIXTracker(self.args)
    #     # ori_thresh = self.args.track_thresh
    #     for cur_iter, (
    #         origin_imgs,
    #         imgs,
    #         inscribed_images,
    #         info_imgs,
    #         ids,
    #     ) in enumerate(progress_bar(self.dataloader)):
    #         # info_imgs is 4 scalar tensors: height, width, frame_id, video_id
    #         with torch.no_grad():
    #             # init tracker
    #             frame_id = info_imgs[2][0]
    #             video_id = info_imgs[3][0].item()
    #             img_file_name = info_imgs[4]
    #             video_name = img_file_name[0].split("/")[-1]
    #             batch_size = imgs.shape[0]

    #             if video_name not in video_names:
    #                 video_names[video_id] = video_name
    #             if frame_id == 1:
    #                 self.args = set_args(self.args, video_name)
    #                 if "MOT17" in video_name:
    #                     self.args.alpha = 0.8778
    #                     self.args.iou_thresh = 0.2217
    #                     self.args.match_thresh = 0.7986
    #                 tracker.re_init(self.args)
    #                 if len(results) != 0:
    #                     result_filename = os.path.join(
    #                         result_folder, "{}.txt".format(video_names[video_id - 1])
    #                     )
    #                     write_results(result_filename, results)
    #                     results = []

    #             imgs = imgs.type(tensor_type)

    #             # skip the the last iters since batchsize might be not enough for batch inference
    #             is_time_record = cur_iter < len(self.dataloader) - 1
    #             if is_time_record:
    #                 start = time.time()

    #             if self.timer is None:
    #                 self.timer = Timer()
    #             self.timer.tic()

    #             self.preproc_timer.tic()
    #             self.preproc_timer_counter += 1

    #             with torch.no_grad():
    #                 outputs = model(imgs)
    #                 # print(outputs)

    #             self.timer.toc()
    #             self.timer_counter += 1
    #             if self.timer_counter % (50 // batch_size) == 0:
    #                 logger.info(
    #                     "Model forward pass {} ({:.2f} fps)".format(
    #                         frame_id,
    #                         (1.0 / max(1e-5, self.timer.average_time) * batch_size),
    #                     )
    #                 )
    #                 self.timer = Timer()

    #             if decoder is not None:
    #                 outputs = decoder(outputs, dtype=outputs.type())

    #             outputs = postprocess(
    #                 outputs, self.num_classes, self.confthre, self.nmsthre
    #             )
    #             if outputs and outputs[0] is not None:
    #                 # print(f" >>> {outputs[0].shape[0]} detections")
    #                 assert outputs[0].shape[1] == 7  # Yolox output has 7 fields?

    #             if is_time_record:
    #                 infer_end = time_synchronized()
    #                 inference_time += infer_end - start

    #         output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
    #         data_list.extend(output_results)

    #         # if outputs[0] is not None:
    #         #     outputs = postprocess(
    #         #         torch.stack(outputs), self.num_classes, self.confthre, self.nmsthre
    #         #     )
    #         frame_count = len(outputs)
    #         for frame_index in range(len(outputs)):
    #             frame_id = info_imgs[2][frame_index]
    #             # print(f"frame_id={frame_id}")
    #             # run tracking
    #             if outputs[frame_index] is not None:
    #                 self.track_timer.tic()
    #                 this_img_info = [
    #                     info_imgs[0][frame_index],
    #                     info_imgs[1][frame_index],
    #                     info_imgs[2][frame_index],
    #                     info_imgs[3],
    #                 ]
    #                 online_targets, detections = tracker.update(
    #                     outputs[frame_index],
    #                     this_img_info,
    #                     self.img_size,
    #                     imgs[frame_index].cuda(),
    #                     # origin_imgs[frame_index].cuda(),
    #                 )
    #                 # continue

    #                 online_tlwhs = []
    #                 online_ids = []
    #                 online_scores = []
    #                 # if online_targets:
    #                 #     print(f"{len(online_targets)} targets, {len(detections)} detections")
    #                 for t in online_targets:
    #                     tlwh = t.tlwh
    #                     tid = t.track_id
    #                     vertical = tlwh[2] / tlwh[3] > 1.6
    #                     if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
    #                         online_tlwhs.append(tlwh)
    #                         online_ids.append(tid)
    #                         online_scores.append(t.score)
    #                     else:
    #                         print("Skipping target")
    #                 self.track_timer.toc()
    #                 self.track_timer_counter += 1
    #                 if self.track_timer_counter % 50 == 0:
    #                     logger.info(
    #                         "Tracking {} ({:.2f} fps)".format(
    #                             frame_id, 1.0 / max(1e-5, self.track_timer.average_time)
    #                         )
    #                     )
    #                     self.track_timer = Timer()
    #                 self.preproc_timer.toc()
    #                 if self.online_callback is not None:
    #                     detections, online_tlwhs = self.online_callback(
    #                         frame_id=frame_id,
    #                         online_tlwhs=online_tlwhs,
    #                         online_ids=online_ids,
    #                         online_scores=online_scores,
    #                         detections=detections,
    #                         info_imgs=info_imgs,
    #                         img=imgs[frame_index].unsqueeze(0),
    #                         inscribed_image=inscribed_images[frame_index].unsqueeze(0),
    #                         original_img=origin_imgs[frame_index].unsqueeze(0),
    #                     )
    #                 if frame_index < frame_count - 1:
    #                     self.preproc_timer.tic()
    #                 # save results
    #                 if isinstance(online_tlwhs, torch.Tensor):
    #                     online_tlwhs = online_tlwhs.numpy()
    #                 if isinstance(online_ids, torch.Tensor):
    #                     online_ids = online_ids.numpy()
    #                 results.append(
    #                     (frame_id.item(), online_tlwhs, online_ids, online_scores)
    #                 )

    #             if is_time_record:
    #                 track_end = time_synchronized()
    #                 track_time += track_end - infer_end

    #             if cur_iter == len(self.dataloader) - 1:
    #                 result_filename = os.path.join(
    #                     result_folder, "{}.txt".format(video_names[video_id])
    #                 )
    #                 write_results(result_filename, results)
    #             # end frame loop
    #         # After frame loop
    #         if self.preproc_timer_counter % 20 == 0:
    #             logger.info(
    #                 ">>> Preproc {} ({:.2f} fps)".format(
    #                     frame_id,
    #                     frame_count * 1.0 / max(1e-5, self.preproc_timer.average_time),
    #                 )
    #             )
    #             self.preproc_timer = Timer()

    #     # always write results
    #     result_filename = os.path.join(
    #         result_folder, "{}.txt".format(video_names[video_id])
    #     )
    #     write_results(result_filename, results)

    #     statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
    #     if distributed:
    #         data_list = gather(data_list, dst=0)
    #         data_list = list(itertools.chain(*data_list))
    #         torch.distributed.reduce(statistics, dst=0)

    #     eval_results = self.evaluate_prediction(data_list, statistics)
    #     synchronize()
    #     return eval_results

    def evaluate_hockeymom(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        evaluate: bool = False,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        if self.args.iou_only:
            # Still use the old onee, I guess
            from yolox.mixsort_tracker.mixsort_iou_tracker import (
                MIXTracker as HMTracker,
            )
        else:
            from yolox.mixsort_tracker.hm_tracker import HMTracker

        # like ByteTrack, we use different setting for different videos
        setting = {
            "MOT17-01-FRCNN": {"track_buffer": 27, "track_thresh": 0.6275},
            "MOT17-03-FRCNN": {"track_buffer": 31, "track_thresh": 0.5722},
            "MOT17-06-FRCNN": {"track_buffer": 16, "track_thresh": 0.5446},
            "MOT17-07-FRCNN": {"track_buffer": 24, "track_thresh": 0.5939},
            "MOT17-08-FRCNN": {"track_buffer": 24, "track_thresh": 0.7449},
            "MOT17-12-FRCNN": {"track_buffer": 29, "track_thresh": 0.7036},
            "MOT17-14-FRCNN": {"track_buffer": 28, "track_thresh": 0.5436},
        }

        def set_args(args, video):
            if video not in setting.keys():
                return args
            for k, v in setting[video].items():
                args.__setattr__(k, v)
            return args

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() and not self.online_callback else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = HMTracker(self.args)
        # ori_thresh = self.args.track_thresh
        for cur_iter, (
            origin_imgs,
            imgs,
            inscribed_images,
            info_imgs,
            ids,
        ) in enumerate(progress_bar(self.dataloader)):
            # info_imgs is 4 scalar tensors: height, width, frame_id, video_id
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2][0]
                video_id = info_imgs[3][0].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[-1]
                batch_size = imgs.shape[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    self.args = set_args(self.args, video_name)
                    if "MOT17" in video_name:
                        self.args.alpha = 0.8778
                        self.args.iou_thresh = 0.2217
                        self.args.match_thresh = 0.7986
                    tracker.re_init(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(
                            result_folder, "{}.txt".format(video_names[video_id - 1])
                        )
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if self.timer is None:
                    self.timer = Timer()
                self.timer.tic()

                self.preproc_timer.tic()
                self.preproc_timer_counter += 1

                with torch.no_grad():
                    outputs = model(imgs)
                    # print(outputs)

                self.timer.toc()
                self.timer_counter += 1
                if self.timer_counter % (50 // batch_size) == 0:
                    logger.info(
                        "Model forward pass {} ({:.2f} fps)".format(
                            frame_id,
                            batch_size * 1.0 / max(1e-5, self.timer.average_time),
                        )
                    )
                    self.timer = Timer()

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if outputs and outputs[0] is not None:
                    assert outputs[0].shape[1] == 7  # Yolox output has 7 fields

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                if self.postprocessor is not None:
                    outputs = (
                        self.dataloader.scale_letterbox_to_original_image_coordinates(
                            outputs,
                        )
                    )

            # outputs[1] = None
            output_results = self.convert_to_coco_format_post_scale(outputs, ids)
            outputs, output_results = self.filter_outputs(outputs, output_results)

            data_list.extend(output_results)

            for frame_index in range(len(outputs)):
                frame_id = info_imgs[2][frame_index]

                online_tlwhs = []
                online_ids = []
                online_scores = []
                detections = []

                # run tracking
                if outputs[frame_index] is not None:
                    self.track_timer.tic()
                    online_targets, detections = tracker.update(
                        outputs[frame_index],
                        origin_imgs[frame_index].cuda(),
                        self.dataloader.dataset.class_ids,
                    )

                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                        else:
                            print("Skipping target")
                else:
                    print(f"No tracking items on frame {frame_id}")

                self.track_timer.toc()
                self.track_timer_counter += 1
                if self.track_timer_counter % 50 == 0:
                    logger.info(
                        "Tracking {} ({:.2f} fps)".format(
                            frame_id, 1.0 / max(1e-5, self.track_timer.average_time)
                        )
                    )
                    self.track_timer = Timer()

                if self.online_callback is not None:
                    detections, online_tlwhs = self.online_callback(
                        frame_id=frame_id,
                        online_tlwhs=online_tlwhs,
                        online_ids=online_ids,
                        online_scores=online_scores,
                        detections=detections,
                        info_imgs=info_imgs,
                        letterbox_img=imgs[frame_index].unsqueeze(0),
                        inscribed_img=inscribed_images[frame_index].unsqueeze(0),
                        original_img=origin_imgs[frame_index].unsqueeze(0),
                    )

                    # save results
                    if isinstance(online_tlwhs, torch.Tensor):
                        online_tlwhs = online_tlwhs.numpy()
                    if isinstance(online_ids, torch.Tensor):
                        online_ids = online_ids.numpy()
                    if online_scores:
                        online_scores = torch.stack(online_scores).numpy()
                    results.append(
                        (frame_id.item(), online_tlwhs, online_ids, online_scores)
                    )

                if is_time_record:
                    track_end = time_synchronized()
                    track_time += track_end - infer_end

                if cur_iter == len(self.dataloader) - 1:
                    result_filename = os.path.join(
                        result_folder, "{}.txt".format(video_names[video_id])
                    )
                    write_results(result_filename, results)

                # end frame loop
            #
            # After frame loop
            #
            self.preproc_timer.toc()
            if self.preproc_timer_counter % 20 == 0:
                logger.info(
                    ">>> Preproc {} ({:.2f} fps)".format(
                        frame_id,
                        batch_size * 1.0 / max(1e-5, self.preproc_timer.average_time),
                    )
                )
                self.preproc_timer = Timer()

        # always write results
        result_filename = os.path.join(
            result_folder, "{}.txt".format(video_names[video_id])
        )
        write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        if not evaluate:
            return data_list, statistics
        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_fair(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        evaluate: bool = False,
        tracker_name="jde",
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """

        # if self.args.iou_only:
        #     # Still use the old onee, I guess
        #     from yolox.mixsort_tracker.mixsort_iou_tracker import (
        #         MIXTracker as HMTracker,
        #     )
        # else:
        #     from yolox.mixsort_tracker.hm_tracker import HMTracker

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        assert model is None
        # model = model.eval()
        # if half:
        #     model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() and not self.online_callback else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        tracker = JDETracker(opt=self.args, frame_rate=self.dataloader.fps)

        for cur_iter, (
            origin_imgs,
            letterbox_imgs,
            inscribed_images,
            info_imgs,
            ids,
        ) in enumerate(progress_bar(self.dataloader)):
            # info_imgs is 4 scalar tensors: height, width, frame_id, video_id
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2][0]
                video_id = info_imgs[3][0].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[-1]
                batch_size = letterbox_imgs.shape[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    if len(results) != 0:
                        result_filename = os.path.join(
                            result_folder, "{}.txt".format(video_names[video_id - 1])
                        )
                        write_results(result_filename, results)
                        results = []

                letterbox_imgs = letterbox_imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if self.timer is None:
                    self.timer = Timer()
                self.timer.tic()

                self.preproc_timer.tic()
                self.preproc_timer_counter += 1

                #assert origin_imgs.shape[0] == 1  # TODO: support batch
                # origin_imgs = origin_imgs.squeeze(0).permute(1, 2, 0).contiguous()
                dets, id_feature = tracker.detect(
                    letterbox_imgs,
                    origin_imgs.permute(0, 2, 3, 1),
                    #origin_imgs.squeeze(0).permute(1, 2, 0),
                )

                # outputs[1] = None
                # output_results = self.convert_to_coco_format_post_scale(outputs, ids)
                # outputs, output_results = self.filter_outputs(outputs, output_results)

                # data_list.extend(output_results)

                for frame_index in range(len(letterbox_imgs)):
                    frame_id = info_imgs[2][frame_index]
                    detections = dets[frame_index]

                    online_targets = tracker.inner_update(
                        detections, id_feature[frame_index]
                    )

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []

                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                            online_tlwhs.append(torch.from_numpy(tlwh))
                            online_ids.append(tid)
                            online_scores.append(t.score)
                        else:
                            print(f"Skipping target: tlwh={tlwh}")

                    if online_ids:
                        online_ids = torch.tensor(online_ids, dtype=torch.int64)
                        online_tlwhs = torch.stack(online_tlwhs)

                    if self.online_callback is not None:
                        detections, online_tlwhs = self.online_callback(
                            frame_id=frame_id,
                            online_tlwhs=online_tlwhs,
                            online_ids=online_ids,
                            online_scores=online_scores,
                            detections=detections,
                            info_imgs=info_imgs,
                            letterbox_img=letterbox_imgs[frame_index].unsqueeze(0),
                            inscribed_img=inscribed_images[frame_index].unsqueeze(0),
                            original_img=origin_imgs[frame_index].unsqueeze(0),
                        )

                        # save results
                        if isinstance(online_tlwhs, torch.Tensor):
                            online_tlwhs = online_tlwhs.numpy()
                        if isinstance(online_ids, torch.Tensor):
                            online_ids = online_ids.numpy()
                        if online_scores and isinstance(online_scores, torch.Tensor):
                            online_scores = torch.stack(online_scores).numpy()
                        results.append(
                            (frame_id.item(), online_tlwhs, online_ids, online_scores)
                        )
                # end frame loop

                # if is_time_record:
                #     track_end = time_synchronized()
                #     track_time += track_end - infer_end

                if cur_iter == len(self.dataloader) - 1:
                    result_filename = os.path.join(
                        result_folder, "{}.txt".format(video_names[video_id])
                    )
                    write_results(result_filename, results)

            #
            # After frame loop
            #
            self.preproc_timer.toc()
            if self.preproc_timer_counter % 20 == 0:
                logger.info(
                    ">>> FairMOT Tracking {} ({:.2f} fps)".format(
                        frame_id,
                        batch_size * 1.0 / max(1e-5, self.preproc_timer.average_time),
                    )
                )
                self.preproc_timer = Timer()

        # always write results
        result_filename = os.path.join(
            result_folder, "{}.txt".format(video_names[video_id])
        )
        write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        if not evaluate:
            return data_list, statistics
        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_mixsort_oc(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        from yolox.mixsort_oc_tracker.mixsort_oc_tracker import MIXTracker

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        seq_data_list = dict()
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = MIXTracker(
            det_thresh=self.args.track_thresh,
            args=self.args,
            iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso,
            delta_t=self.args.deltat,
            inertia=self.args.inertia,
            use_byte=self.args.use_byte,
            max_age=self.args.track_buffer,
        )
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                frame_id = info_imgs[2].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[0]
                if frame_id == 1:
                    tracker.re_init()
                    if len(results) != 0:
                        result_filename = os.path.join(
                            result_folder, "{}.txt".format(video_names[video_id])
                        )
                        write_results_no_score(result_filename, results)
                        results = []
                # init tracker
                video_id = info_imgs[3].item()
                img_name = img_file_name[0].split("/")[2]
                """
                    Here, you can use adaptive detection threshold as in BYTE
                    (line 268 - 292), which can boost the performance on MOT17/MOT20
                    datasets, but we don't use that by default for a generalized
                    stack of parameters on all datasets.
                """
                if video_name not in video_names:
                    video_names[video_id] = video_name

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if video_name not in seq_data_list:
                seq_data_list[video_name] = []
            seq_data_list[video_name].extend(output_results)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda()
                )
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)

                if self.online_callback is not None:
                    _, online_tlwhs = self.online_callback(
                        frame_id=frame_id,
                        online_tlwhs=online_tlwhs,
                        online_ids=online_ids,
                        online_scores=[],
                        info_imgs=info_imgs,
                        img=imgs,
                        original_img=origin_imgs,
                    )

                # save results
                results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(
                    result_folder, "{}.txt".format(video_names[video_id])
                )
                write_results_no_score(result_filename, results)

        # always write results
        result_filename = os.path.join(
            result_folder, "{}.txt".format(video_names[video_id])
        )
        write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_ocsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        from yolox.ocsort_tracker.ocsort import OCSort

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        seq_data_list = dict()
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = OCSort(
            det_thresh=self.args.track_thresh,
            iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso,
            delta_t=self.args.deltat,
            inertia=self.args.inertia,
        )
        ori_thresh = self.args.track_thresh
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[0]
                img_name = img_file_name[0].split("/")[2]
                """
                    Here, you can use adaptive detection threshold as in BYTE
                    (line 268 - 292), which can boost the performance on MOT17/MOT20
                    datasets, but we don't use that by default for a generalized
                    stack of parameters on all datasets.
                """
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OCSort(
                        det_thresh=self.args.track_thresh,
                        iou_threshold=self.args.iou_thresh,
                        asso_func=self.args.asso,
                        delta_t=self.args.deltat,
                        inertia=self.args.inertia,
                    )
                    if len(results) != 0:
                        try:
                            result_filename = os.path.join(
                                result_folder,
                                "{}.txt".format(video_names[video_id - 1]),
                            )
                        except:
                            import pdb

                            pdb.set_trace()
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)

            if video_name not in seq_data_list:
                seq_data_list[video_name] = []
            seq_data_list[video_name].extend(output_results)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0], info_imgs, self.img_size, origin_imgs.squeeze(0).cuda()
                )
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)

                if self.online_callback is not None:
                    _, online_tlwhs = self.online_callback(
                        frame_id=frame_id,
                        online_tlwhs=online_tlwhs,
                        online_ids=online_ids,
                        online_scores=[],
                        info_imgs=info_imgs,
                        img=imgs,
                        original_img=origin_imgs,
                    )

                # save results
                results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(
                    result_folder, "{}.txt".format(video_names[video_id])
                )
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        # for video_name in seq_data_list.keys():
        #     self.save_detection_result(seq_data_list[video_name], result_folder, video_name)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = Sort(self.args.track_thresh)

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(
                            result_folder, "{}.txt".format(video_names[video_id - 1])
                        )
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            if self.online_callback is not None:
                _, online_tlwhs = self.online_callback(
                    frame_id=frame_id,
                    online_tlwhs=online_tlwhs,
                    online_ids=online_ids,
                    online_scores=[],
                    info_imgs=info_imgs,
                    img=imgs,
                    original_img=imgs,
                )

            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(
                    result_folder, "{}.txt".format(video_names[video_id])
                )
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(
                        model_folder, min_confidence=self.args.track_thresh
                    )
                    if len(results) != 0:
                        result_filename = os.path.join(
                            result_folder, "{}.txt".format(video_names[video_id - 1])
                        )
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(
                outputs[0], info_imgs, self.img_size, img_file_name[0]
            )
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            if self.online_callback is not None:
                _, online_tlwhs = self.online_callback(
                    frame_id=frame_id,
                    online_tlwhs=online_tlwhs,
                    online_ids=online_ids,
                    online_scores=[],
                    info_imgs=info_imgs,
                    img=imgs,
                    original_img=imgs,
                )

            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(
                    result_folder, "{}.txt".format(video_names[video_id])
                )
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(
                        model_folder, min_cls_score=self.args.track_thresh
                    )
                    if len(results) != 0:
                        result_filename = os.path.join(
                            result_folder, "{}.txt".format(video_names[video_id - 1])
                        )
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(
                outputs[0], info_imgs, self.img_size, img_file_name[0]
            )
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            if self.online_callback is not None:
                _, online_tlwhs = self.online_callback(
                    frame_id=frame_id,
                    online_tlwhs=online_tlwhs,
                    online_ids=online_ids,
                    online_scores=online_scores,
                    info_imgs=info_imgs,
                    img=imgs,
                    original_img=imgs,
                )

            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(
                    result_folder, "{}.txt".format(video_names[video_id])
                )
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format_post_scale(self, outputs, ids):
        data_list = []
        for output, img_id in zip(outputs, ids):
            if output is None:
                data_list.append(None)
                continue
            output = output.cpu()
            bboxes = output[:, 0:4]

            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for output, img_h, img_w, img_id in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            """
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            """
            # from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
