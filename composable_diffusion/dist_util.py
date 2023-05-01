"""
Helpers for distributed training.
"""

import io
import os
import socket
import subprocess

import blobfile as bf
import torch as th
import torch.distributed as dist


def setup_dist(backend="nccl"):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    dist.init_process_group(backend=backend, init_method="env://")

    if th.cuda.is_available():  # This clears remaining caches in GPU 0
        th.cuda.set_device(dev())
        th.cuda.empty_cache()


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            # Create a copy of the parameter tensor to avoid broadcasting failures
            p_copy = p.detach().clone()
            dist.broadcast(p_copy, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
