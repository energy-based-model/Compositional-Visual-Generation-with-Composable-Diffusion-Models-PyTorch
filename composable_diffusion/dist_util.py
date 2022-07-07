# """
# Helpers for distributed training.
# """
#
# import io
# import os
# import socket
#
# import blobfile as bf
# from mpi4py import MPI
# import torch as th
# import torch.distributed as dist
#
# # Change this to reflect your cluster layout.
# # The GPU for a given rank is (rank % GPUS_PER_NODE).
# GPUS_PER_NODE = 2
#
# SETUP_RETRY_COUNT = 3
#
#
# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return
#
#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"
#
#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)
#
#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")
#
#
# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if th.cuda.is_available():
#         return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
#     return th.device("cpu")
#
#
# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     if MPI.COMM_WORLD.Get_rank() == 0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#     else:
#         data = None
#     data = MPI.COMM_WORLD.bcast(data)
#     return th.load(io.BytesIO(data), **kwargs)
#
#
# def sync_params(params):
#     """
#     Synchronize a sequence of Tensors across ranks from rank 0.
#     """
#     for p in params:
#         with th.no_grad():
#             dist.broadcast(p, 0)
#
#
# def _find_free_port():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(("", 0))
#         s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         return s.getsockname()[1]
#     finally:
#         s.close()


"""
Helpers for distributed training.
"""

import io
import os
import socket
import subprocess

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    SLURM_VARIABLES = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_TASKS_PER_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_NODEID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_TASK_PID",
    ]

    PREFIX = "%i - " % int(os.environ["SLURM_PROCID"])
    for name in SLURM_VARIABLES:
        value = os.environ.get(name, None)
        print(PREFIX + "%s: %s" % (name, str(value)))

    # number of nodes / node ID
    params.nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    params.node_rank = int(os.environ["SLURM_NODEID"])

    # define master address and master port
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    params.master_addr = hostnames.split()[0].decode("utf-8")
    print("master address ", params.master_addr)