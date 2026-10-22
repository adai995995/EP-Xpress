"""CUDA correctness tests; no device or toolkit means a clean skip."""

import ctypes
from contextlib import contextmanager
import mmap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ep_xpress import native

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.fixture(scope="module", autouse=True)
def native_extension():
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").exists():
        pytest.skip("The CUDA toolkit is unavailable")
    return native.load()


@contextmanager
def registered_mapping(nbytes=4096):
    mapping = mmap.mmap(-1, nbytes)
    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    native.register_host(address, nbytes)
    try:
        yield mapping, address
    finally:
        native.unregister_host(address)
        mapping.close()


def test_add_partial_accumulates_unique_rows():
    generator = torch.Generator().manual_seed(73)
    initial = torch.randn(7, 513, generator=generator, dtype=torch.float32)
    first = torch.randn(3, 513, generator=generator).bfloat16()
    second = torch.randn(2, 513, generator=generator).bfloat16()
    first_indices = torch.tensor([5, 0, 3], dtype=torch.int64)
    second_indices = torch.tensor([0, 6], dtype=torch.int64)
    expected = initial.clone()
    expected[first_indices] += first.float()
    expected[second_indices] += second.float()
    accum = initial.cuda()
    native.add_partial(accum, first.cuda(), first_indices.cuda())
    native.add_partial(accum, second.cuda(), second_indices.cuda())
    torch.testing.assert_close(accum.cpu(), expected, rtol=0, atol=0)


def test_finish_combine_preserves_intermediate_bf16_rounding():
    remote = torch.tensor([[1.0034, -1.0034, 257.25, 0.125]], device="cuda")
    local = torch.tensor([[-1.0, 1.0, -256.0, 0.25]], device="cuda", dtype=torch.bfloat16)
    result = native.finish_combine(remote, local)
    expected = remote.bfloat16() + local
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # The first element would differ if the intermediate rounding were omitted.
    assert result[0, 0].item() != (remote + local.float()).bfloat16()[0, 0].item()


@pytest.mark.parametrize("tokens,hidden", [(0, 32), (4, 0)])
def test_empty_tensors(tokens, hidden):
    accum = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
    partial = torch.empty(0, hidden, device="cuda", dtype=torch.bfloat16)
    indices = torch.empty(0, device="cuda", dtype=torch.int64)
    native.add_partial(accum, partial, indices)
    local = torch.zeros_like(accum, dtype=torch.bfloat16)
    result = native.finish_combine(accum, local)
    assert result.shape == local.shape
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("indices", [[1, 1], [-1, 1], [0, 4]])
def test_invalid_token_indices_are_rejected(indices):
    accum = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    partial = torch.zeros(2, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        native.add_partial(accum, partial, torch.tensor(indices, device="cuda"))


def test_operator_metadata_is_checked():
    accum = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    partial = torch.zeros(2, 8, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
    with pytest.raises(TypeError, match="dtype"):
        native.add_partial(accum.bfloat16(), partial, indices)
    with pytest.raises(TypeError, match="int64"):
        native.add_partial(accum, partial, indices.int())
    with pytest.raises(ValueError, match="hidden dimensions"):
        native.add_partial(accum, partial[:, :4].contiguous(), indices)
    with pytest.raises(ValueError, match="contiguous"):
        native.add_partial(accum, partial[:, ::2], indices)
    with pytest.raises(ValueError, match="length"):
        native.add_partial(accum, partial, indices[:1])
    with pytest.raises(ValueError, match="same CUDA device"):
        native.add_partial(accum, partial, indices.cpu())
    with pytest.raises(ValueError, match="shape and device"):
        native.finish_combine(accum, partial)


def test_operators_follow_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        accum = torch.zeros(4, 17, device="cuda")
        partial = torch.full((2, 17), 2.0, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor([1, 3], device="cuda")
        local = torch.full((4, 17), 0.5, device="cuda", dtype=torch.bfloat16)
        native.add_partial(accum, partial, indices, validate_indices=False)
        result = native.finish_combine(accum, local)
        completion = stream.record_event()
    completion.synchronize()
    expected = torch.full((4, 17), 0.5, dtype=torch.bfloat16)
    expected[[1, 3]] = 2.5
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_registered_host_copy_and_round_trip():
    source = torch.arange(48, device="cuda", dtype=torch.float32).reshape(6, 8).bfloat16()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with registered_mapping() as (mapping, address):
        host = torch.frombuffer(mapping, dtype=torch.bfloat16, count=48).reshape(6, 8)
        host.fill_(-9)
        native.copy_d2h_async(address + 16, source, 2, 3, stream.cuda_stream)
        ready = stream.record_event()
        ready.synchronize()
        torch.testing.assert_close(host[1:4], source[2:5].cpu(), rtol=0, atol=0)
        assert torch.all(host[0] == -9)
        assert torch.all(host[4:] == -9)
        with torch.cuda.stream(stream):
            device_copy = torch.empty((6, 8), device="cuda", dtype=torch.bfloat16)
            device_copy.copy_(host, non_blocking=True)
            done = stream.record_event()
        done.synchronize()
        torch.testing.assert_close(device_copy.cpu(), host, rtol=0, atol=0)
        del host


def test_host_registration_and_copy_bounds():
    source = torch.zeros(4, 8, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    with registered_mapping() as (_, address):
        with pytest.raises(RuntimeError, match="overlaps"):
            native.register_host(address, 4096)
        with pytest.raises(RuntimeError, match="outside"):
            native.copy_d2h_async(address + 4090, source, 0, 1, stream.cuda_stream)
        with pytest.raises(ValueError, match="row slice"):
            native.copy_d2h_async(address, source, 3, 2, stream.cuda_stream)
        with pytest.raises(ValueError, match="row_start"):
            native.copy_d2h_async(address, source, -1, 1, stream.cuda_stream)
        native.copy_d2h_async(address, source, 4, 0, stream.cuda_stream)
    with pytest.raises(RuntimeError, match="not registered"):
        native.unregister_host(address)


def test_validated_native_operators_support_graph_capture():
    accum = torch.zeros(3, 8, device="cuda", dtype=torch.float32)
    partial = torch.full((2, 8), 1.5, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([0, 2], device="cuda")
    local = torch.full((3, 8), 0.5, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        native.add_partial(accum, partial, indices, validate_indices=False)
        native.finish_combine(accum, local)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        accum.zero_()
        native.add_partial(accum, partial, indices, validate_indices=False)
        result = native.finish_combine(accum, local)
    graph.replay()
    expected = torch.full((3, 8), 0.5, dtype=torch.bfloat16)
    expected[[0, 2]] = 2.0
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
