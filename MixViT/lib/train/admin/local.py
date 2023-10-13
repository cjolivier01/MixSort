import os


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = "."  # Base directory for saving network checkpoints.
        self.dataset_dir = f"{os.path.join(os.environ['HOME'], 'src', 'datasets')}"
        self.tensorboard_dir = (
            self.workspace_dir + "/tensorboard/"
        )  # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + "/pretrained_networks/"
        self.sportsmot_dir = f"{self.dataset_dir}/SportsMOT"
        self.sportsmot_anno_dir = f"{self.dataset_dir}/SportsMOT/tracking_annos"
        self.mot17_dir = f"{self.dataset_dir}/MOT17"
        self.mot17_anno_dir = f"{self.dataset_dir}/MOT17/tracking_annos"
        self.mot20_dir = ""
        self.mot20_anno_dir = ""
        self.dancetrack_dir = f"{self.dataset_dir}/DanceTrack"
        self.dancetrack_anno_dir = f"{self.dataset_dir}/DanceTrack/tracking_annos"
        self.soccernet_dir = f"{self.dataset_dir}/SoccerNet"
        self.soccernet_anno_dir = f"{self.dataset_dir}/SoccerNet/tracking_annos"
        self.lasot_dir = ""
        self.tnl2k_dir = ""
        self.got10k_dir = ""
        self.trackingnet_dir = ""
        self.coco_dir = ""
        self.lvis_dir = ""
        self.sbd_dir = ""
        self.imagenet_dir = ""
        self.imagenetdet_dir = ""
        self.ecssd_dir = ""
        self.hkuis_dir = ""
        self.msra10k_dir = ""
        self.davis_dir = ""
        self.youtubevos_dir = ""


if __name__ == "__main__":
    print(EnvironmentSettings())
