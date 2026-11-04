"""Borrowed NCCL lifetime, subgroup ordering and host-SM integration."""

import tempfile
import gc
import weakref
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
from ep_xpress import Config, Context, get_nccl_comm_handle, native

pytestmark = [pytest.mark.cuda, pytest.mark.distributed, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="three CUDA GPUs are required for non-contiguous subgroup coverage")]


def _worker(global_rank, init_file):
    import torch.distributed as dist

    torch.cuda.set_device(global_rank)
    # The application owns the only communicator backend: NCCL.
    dist.init_process_group("nccl", init_method="file://" + init_file, rank=global_rank, world_size=3)
    group = dist.new_group(ranks=[0, 2], backend="nccl")
    if global_rank in (0, 2):
        rank = dist.get_rank(group)
        handle = get_nccl_comm_handle(group)
        assert get_nccl_comm_handle(group) is handle
        assert (handle.rank, handle.world_size, handle.device.index) == (rank, 2, global_rank)
        assert handle.get() > 0
        pointer, reference = handle.get(), weakref.ref(handle)
        del handle
        gc.collect()
        assert reference() is None
        # Ranks may retain handles for different Python lifetimes. Releasing
        # one must not introduce another collective on only that rank.
        with patch.object(dist, "all_reduce", side_effect=AssertionError("unexpected repeated warmup")):
            handle = get_nccl_comm_handle(group)
        assert handle.get() == pointer
        message = {"rank": rank, "routes": ([], [0, -1], None), "blob": b"\x00\xff"}
        gathered = handle.all_gather_object(message)
        assert gathered == [dict(message, rank=r) for r in range(2)]
        config = Config(128, 2, 8, transport="host_sm", channels=2,
                        chunk_bytes=4096, decode_max_tokens=8, fatal_errors=True)
        # Repeated contexts reuse the same upper-layer handle without new groups.
        for _ in range(2):
            with patch.object(dist, "new_group", side_effect=AssertionError("unexpected internal group")):
                context = Context(handle, [0, 0, 1, 1], config)
            assert context.nccl_comm_handle is handle
            try:
                for step in range(3):
                    n = (rank + step) % 3  # zero rows, uneven counts, changing destinations
                    x = torch.full((n, 128), (rank + step + 1) * 8,
                                   device="cuda", dtype=torch.bfloat16)
                    patterns = [[0, 1], [0, 2], [2, 3], [-1, -1]]
                    ids = torch.tensor([patterns[(i + step + rank) % 4] for i in range(n)],
                                       device="cuda", dtype=torch.int64).reshape(n, 2)
                    weights = torch.tensor([[0.25, 0.75]] * n, device="cuda").reshape(n, 2)
                    batch = context.dispatch_async(x, ids, weights).wait()
                    partial = (batch.hidden.float() *
                               (batch.weights * (batch.local_ids >= 0)).sum(1)[:, None]).bfloat16()
                    output = context.combine_async(partial, batch.handle).wait()
                    expected = x.float() * (weights * (ids >= 0)).sum(1)[:, None]
                    torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)
            finally:
                with patch.object(dist, "destroy_process_group", side_effect=AssertionError("borrowed group destroyed")):
                    context.close()
            assert context.nccl_comm_handle is None
            # Closing or failing a Context must not invalidate its upstream comm.
            value = torch.tensor([rank + 1], device="cuda")
            dist.all_reduce(value, group=group)
            assert value.item() == 3
            assert handle.all_gather_object(rank) == [0, 1]
        # Same device and membership guards happen before allocating/compiling.
        with pytest.raises(ValueError, match="same CUDA device"):
            Context(handle, [0, 1], config, device=1)
        handle.barrier()
        del handle
        dist.destroy_process_group(group)
    dist.barrier()
    dist.destroy_process_group()


def test_nccl_handle_subgroup_dynamic_routes_and_borrowed_lifetime():
    native.load()
    with tempfile.TemporaryDirectory(prefix="ep-xpress-nccl-test-") as root:
        torch.multiprocessing.spawn(_worker, args=(root + "/init",), nprocs=3, join=True)
