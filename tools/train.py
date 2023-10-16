import os
from loguru import logger

import socket
import torch
import re
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp

import argparse
import random
import warnings


def get_first_hostname(nodelist):
    """Extract the first hostname from a SLURM nodelist string."""
    # Extract the root hostname and range (if present)
    match = re.match(r"([\w\-]+?)\[(\d+-\d+)?\]", nodelist)
    if match:
        prefix, rng = match.groups()
        if rng:
            # If a range is present, extract the first number
            start = int(rng.split("-")[0])
        else:
            start = None
    else:
        # If no range is present, the nodelist is the hostname
        prefix = nodelist
        start = None

    # Construct the first hostname
    if start is not None:
        return f"{prefix}{start}"
    else:
        return prefix


def get_dist_url(hostname, port=29500, protocol="tcp"):
    """Generate a PyTorch dist-url using the given hostname and port."""
    ip = socket.gethostbyname(hostname)
    os.environ["MASTER_PORT"] = f"{port}"
    os.environ["MASTER_ADDR"] = ip
    return f"{protocol}://{ip}:{port}"


def get_default_dist_url():
    nodelist = os.environ.get("SLURM_JOB_NODELIST", None)
    if not nodelist:
        return None
    master = get_first_hostname(nodelist)
    if master:
        return get_dist_url(master)
    return None


def get_local_rank():
    lr = int(os.environ.get("SLURM_LOCALID", "0"))
    os.environ["LOCAL_RANK"] = str(lr)
    return lr


def get_devices():
    if "SLURM_LOCALID" in os.environ:
        return get_local_rank()
    return None


def get_machine_rank():
    return int(os.environ.get("SLURM_PROCID", "0"))


def get_world_size():
    return int(os.environ.get("SLURM_NTASKS", "1"))


def get_dist_backend():
    return "nccl"
    # return "gloo"


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend",
        default=get_dist_backend(),
        type=str,
        help="distributed backend",
    )
    parser.add_argument(
        "--dist-url",
        default=get_default_dist_url(),
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank",
        default=get_local_rank(),
        type=int,
        help="local rank for dist training",
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines",
        default=get_world_size(),
        type=int,
        help="num of node for training",
    )
    parser.add_argument(
        "--machine_rank",
        default=get_machine_rank(),
        type=int,
        help="node rank for multi-node training",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    print(f"num_gpu={num_gpu}")
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args),
    )
