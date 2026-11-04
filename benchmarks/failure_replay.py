"""Two-rank metadata validation checks on a healthy borrowed NCCL communicator.

Missing-rank NCCL startup/metadata failures require an external worker deadline;
the raw borrowed communicator has no EP-Xpress-owned watchdog or abort path.
"""

import argparse
from datetime import timedelta
import json
import os
import time

import torch
import torch.distributed as dist

from ep_xpress import get_nccl_comm_handle, Config, Context
from ep_xpress import native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    native.load()
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    if dist.get_world_size() != 2:
        raise ValueError("failure replay requires two ranks")
    config = Config(16, 1, 4, chunk_bytes=64, timeout_seconds=args.timeout)
    x = torch.ones((2, 16), dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros((2, 1), dtype=torch.int64, device="cuda")
    weights = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    results = []
    for case in ("tag_mismatch", "invalid_route"):
        context = Context(get_nccl_comm_handle(dist.group.WORLD), [0, 1], config)
        start = time.monotonic()
        caught = False
        try:
            selected_ids = ids
            if case == "invalid_route" and rank == 1:
                selected_ids = torch.full_like(ids, 99)
            tag = rank if case == "tag_mismatch" else None
            try:
                context.dispatch_async(x, selected_ids, weights, tag=tag).wait(
                    timeout=args.timeout + 10)
            except (ValueError, RuntimeError) as error:
                expected = {"tag_mismatch": "inconsistent operation order or tags",
                            "invalid_route": "expert ID is outside the supported range"}
                if expected[case] not in str(error):
                    raise AssertionError("failure replay caught an unrelated error") from error
                caught = True
        finally:
            context.close()
        elapsed = time.monotonic() - start
        if not caught or elapsed > args.timeout + 10:
            raise AssertionError("failure did not propagate within the expected bound")
        dist.barrier()
        results.append(case)
    if rank == 0:
        print(json.dumps({"status": "passed", "failure_cases": results}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
