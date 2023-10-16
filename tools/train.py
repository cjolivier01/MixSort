import os
from loguru import logger

import time
import socket
import torch
import re
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp

import argparse
import random
import warnings

def _print_slurm_environment():
    if 'SLURM_PROCID' in os.environ and int(os.environ['SLURM_PROCID']) == 0:
        for key, val in os.environ.items():
            if key.startswith("SLURM_"):
                print(f"{key}={val}")

#_print_slurm_environment()
#exit(0)

node_lists = [
    "clip-g1-[01-03]",
    "clip-g1-[0-1],clip-g2-[2-3]",
    "clip-g1-0,clip-g2-0",
    "clip-g1-0,clip-g2-1",
    "clip-g1-1",
    "clip-a-[1,3,5]",
    "clip-b-[1-3,5]",
    "clip-c-[1-3,5,9-12]",
    "clip-d-[5,9-12]",
    "clip-e-[5,9],clip-e-[15-19]",
    "clip-f-[5,9],clip-f-[15,17]",
    "clip-f-5,clip-f-[15,17]",
    "clip-f-[5,9],clip-f-175",
]


def set_slurm_env_variables(node_list, tasks_per_node):
    """
    Set virtual Slurm environment variables based on the node list and tasks per node.
    
    Args:
    - node_list (list of str): List of node names
    - tasks_per_node (int): Number of tasks per node
    """
    # Setting SLURM_JOB_NODELIST and SLURM_TASKS_PER_NODE
    os.environ['SLURM_JOB_NODELIST'] = ','.join(node_list)
    os.environ['SLURM_TASKS_PER_NODE'] = str(tasks_per_node)

    # Setting SLURM_JOB_NUM_NODES
    os.environ['SLURM_JOB_NUM_NODES'] = str(len(node_list))

    # Setting SLURM_NNODES to be the same as SLURM_JOB_NUM_NODES (for some scripts)
    os.environ['SLURM_NNODES'] = os.environ['SLURM_JOB_NUM_NODES']

    # Setting SLURM_NTASKS as total number of tasks
    os.environ['SLURM_NTASKS'] = str(len(node_list) * tasks_per_node)

    # Mock other possible Slurm environment variables as needed...

def add_string_numbers(a: str, b: str) -> str:
    # Step 2: Make the strings of equal length by prefixing zeros
    max_len = max(len(a), len(b))
    a = a.rjust(max_len, '0')
    b = b.rjust(max_len, '0')

    result = []
    carry = 0

    # Step 3: Start adding from the rightmost digit
    for i in range(max_len - 1, -1, -1):
        total = carry + int(a[i]) + int(b[i])
        carry = total // 10
        result.append(str(total % 10))

    # Step 4: If there's any carry left, add it to the front of the result
    if carry:
        result.append(str(carry))

    # Combine the result and reverse to get the proper order
    return ''.join(result[::-1])


def slurm_parse_int(s):
    for i, c in enumerate(s):
        if c not in "0123456789":
            return s[:i], s[i:]
    return int(s), ""


def string_range(a, b):
    results = [a]
    next_num = a
    while int(next_num) + 1 != int(b):
        next_num = add_string_numbers(next_num, "1")
        results.append(next_num)
    return results

def slurm_parse_brackets(s):
    # parse a "bracket" expression (including closing ']')
    lst = []
    while len(s) > 0:
        if s[0] == ",":
            s = s[1:]
            continue
        if s[0] == "]":
            return lst, s[1:]
        a, s = slurm_parse_int(s)
        assert len(s) > 0, f"Missing closing ']'"
        if s[0] in ",]":
            lst.append(a)
        elif s[0] == "-":
            b, s = slurm_parse_int(s[1:])
            lst += string_range(a, int(b) + 1)
    assert len(s) > 0, f"Missing closing ']'"


def slurm_parse_node(s):
    # parse a "node" expression
    for i, c in enumerate(s):
        if c == ",":  # name,...
            return [s[:i]], s[i + 1 :]
        if c == "[":  # name[v],...
            b, rest = slurm_parse_brackets(s[i + 1 :])
            if len(rest) > 0:
                assert rest[0] == ",", f"Expected comma after brackets in {s[i:]}"
                rest = rest[1:]
            return [s[:i] + str(z) for z in b], rest

    return [s], ""


def slurm_parse_list(s):
    lst = []
    while len(s) > 0:
        v, s = slurm_parse_node(s)
        lst.extend(v)
    return lst


# for s in node_lists:
#     print(s)
#     print(slurm_parse_list(s))

def get_first_hostname(nodelist):
    nodelist = slurm_parse_list(os.environ.get("SLURM_STEP_NODELIST", ""))
    if not nodelist:
        return None
    print(f"Master: {nodelist[0]}")
    return nodelist[0]


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


def get_machine_rank():
    return int(os.environ.get("SLURM_NODEID", "0"))


def get_num_machines():
    num_machines = int(os.environ.get("SLURM_NNODES", "1"))
    os.environ["WORLD_SIZE"] = str(num_machines)
    return num_machines


def get_dist_backend():
    # return "nccl"
    return "gloo"


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
        default=get_num_machines(),
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
    num_gpu = min(num_gpu, 4)
    assert num_gpu <= torch.cuda.device_count()
    time.sleep(args.machine_rank)
    #args.num_machines = 4
    print(f"machine rank: {args.machine_rank}, hostname={socket.gethostname()}, ngpu={num_gpu}, dist_url={args.dist_url}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args),
    )
