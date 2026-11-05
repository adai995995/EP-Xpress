"""Lazy NCCL device-API extension. No communicator is created here."""

import hashlib
import os
from pathlib import Path
import threading

import torch

_extension = None
_lock = threading.Lock()


def load(verbose=False):
    global _extension
    with _lock:
        if _extension is not None:
            return _extension
        from torch.utils.cpp_extension import load as build

        root = Path(os.environ.get("EP_XPRESS_NCCL_ROOT",
                    str(Path(torch.__file__).resolve().parent.parent / "nvidia/nccl")))
        header = root / "include/nccl.h"
        if not header.is_file() or not (root / "include/nccl_device.h").is_file():
            raise RuntimeError("GIN needs NCCL device headers; set EP_XPRESS_NCCL_ROOT")
        source = Path(__file__).resolve().parent / "csrc/gin.cu"
        digest = hashlib.sha256(source.read_bytes() + header.read_bytes()
                                + torch.__version__.encode()).hexdigest()[:12]
        os.environ.setdefault("MAX_JOBS", "1")
        _extension = build(name="ep_xpress_gin_" + digest, sources=[str(source)],
            extra_include_paths=[str(root / "include")],
            extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"],
            extra_ldflags=[f"-L{root / 'lib'}", "-l:libnccl.so.2", f"-Wl,-rpath,{root / 'lib'}"],
            verbose=verbose)
        return _extension
