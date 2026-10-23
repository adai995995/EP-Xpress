"""Focused GPU numerical check for the patched SGLang count-aware BF16 path.

Run explicitly in the pinned, patched SGLang environment. This is not an
end-to-end performance benchmark and does not initialize distributed groups.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as fm
from ep_xpress.adapters.sglang_valid import align_valid_routes


def check_alignment_layout(device):
    """Check stable expert grouping, padding, and the device-count boundary."""
    results = []
    # Exercise both sides of the 1024-route fast-path boundary, tile tails,
    # non-power-of-two expert/tile counts, and a mostly empty large workspace.
    # Inactive capacity rows intentionally contain valid expert IDs: Count,
    # rather than pre-cleared input storage, must bound the work.
    cases = (
        (1, 0, 8, 16, 32, torch.int32, "mixed"),
        (1, 1, 8, 16, 64, torch.int64, "skewed"),
        (127, 83, 8, 7, 64, torch.int32, "mixed"),
        (128, 128, 8, 16, 32, torch.int64, "mixed"),
        (129, 0, 8, 16, 64, torch.int64, "mixed"),
        (129, 83, 8, 16, 32, torch.int32, "mixed"),
        (129, 129, 8, 16, 64, torch.int64, "mixed"),
        (259, 259, 8, 16, 32, torch.int32, "skewed"),
        (513, 509, 8, 7, 128, torch.int64, "mixed"),
        (1025, 1025, 8, 16, 32, torch.int32, "invalid"),
        (4097, 4097, 8, 16, 64, torch.int64, "mixed"),
        (4097, 25, 8, 16, 32, torch.int32, "mixed"),
        (16384, 14763, 8, 16, 64, torch.int64, "mixed"),
        (32769, 32769, 8, 16, 32, torch.int32, "mixed"),
        (1025, 1023, 1, 1, 32, torch.int32, "skewed"),
    )
    for capacity, valid, topk, experts, block_size, dtype, pattern in cases:
        flat = torch.arange(capacity * topk, dtype=torch.int64)
        ids = ((flat * 5 + (flat // topk) * 3) % experts).reshape(capacity, topk)
        ids.reshape(-1)[flat % 7 == 0] = -1
        ids.reshape(-1)[flat % 31 == 0] = experts + 13
        if dtype == torch.int64:
            # Validate the original 64-bit ID before narrowing a histogram key.
            # These values would become valid expert IDs after an int32 cast.
            wide = flat % 29 == 0
            ids.reshape(-1)[wide] = 2**32 + (flat[wide] % experts)
        if pattern == "skewed":
            ids[:valid] = experts - 1
        elif pattern == "invalid":
            ids[:valid] = torch.where(
                flat[:valid * topk].reshape(valid, topk) % 2 == 0,
                -1, experts + 13,
            )
        # Make every inactive row route somewhere if Count were ignored.
        ids[valid:] = experts - 1
        ids = ids.to(dtype)
        sorted_ids, expert_ids, padded = align_valid_routes(
            ids.to(device),
            torch.tensor([valid], device=device, dtype=torch.int32),
            block_size,
            experts,
        )
        maximum = capacity * topk + experts * (block_size - 1)
        assert sorted_ids.shape == (maximum,)
        assert expert_ids.shape == ((maximum + block_size - 1) // block_size,)
        assert sorted_ids.dtype == expert_ids.dtype == padded.dtype == torch.int32
        expected_rows, expected_experts = [], []
        for expert in range(experts):
            positions = (ids[:valid].reshape(-1) == expert).nonzero().flatten().tolist()
            padded_count = ((len(positions) + block_size - 1) // block_size) * block_size
            expected_rows.extend(positions)
            expected_rows.extend([capacity * topk] * (padded_count - len(positions)))
            expected_experts.extend([expert] * (padded_count // block_size))
        # These host observations are confined to this explicit test script.
        padded_rows = int(padded.item())
        assert padded_rows == len(expected_rows)
        torch.testing.assert_close(
            sorted_ids[:padded_rows].cpu(),
            torch.tensor(expected_rows, dtype=torch.int32),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            expert_ids[:padded_rows // block_size].cpu(),
            torch.tensor(expected_experts, dtype=torch.int32),
            rtol=0, atol=0,
        )
        results.append({
            "capacity": capacity, "valid": valid, "topk": topk,
            "experts": experts, "block_size": block_size,
            "ids_dtype": str(dtype), "pattern": pattern,
            "padded_routes": padded_rows, "stable_layout_equal": True,
        })
    return results


def main():
    torch.manual_seed(173)
    device = "cuda"
    alignment_results = check_alignment_layout(device)
    hidden, intermediate, experts, topk = 2048, 768, 16, 8
    w1 = torch.randn((experts, intermediate * 2, hidden), device=device, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn((experts, hidden, intermediate), device=device, dtype=torch.bfloat16) * 0.02
    gemm_config = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                       GROUP_SIZE_M=8, num_warps=4, num_stages=3)
    exec_config = SimpleNamespace(moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False))
    results = []
    with patch.object(fm, "get_exec", lambda: exec_config), \
         patch.object(fm, "_moe_support_tma", lambda: False), \
         patch.object(fm, "is_batch_invariant_mode_enabled", lambda: False), \
         patch.object(fm, "try_get_optimal_moe_config", lambda *a, **k: (dict(gemm_config), (None, None))):
        # Small and large alignment paths; include
        # all-invalid local routes, padding, full capacity, and zero valid rows.
        for capacity, valid in ((1, 0), (1, 1), (32, 1), (32, 19), (32, 32),
                                (129, 0), (129, 83), (129, 129), (256, 256)):
            x = torch.randn((capacity, hidden), device=device, dtype=torch.bfloat16)
            ids = torch.rand((capacity, 128), device=device).argsort(dim=1)[:, :topk]
            ids = torch.where(ids < experts, ids, -1).to(torch.int64).contiguous()
            weights = torch.rand((capacity, topk), device=device, dtype=torch.float32)
            ids[valid:] = -1
            weights[valid:] = 0
            count = torch.tensor([valid], device=device, dtype=torch.int32)
            sorted_ids, expert_ids, padded = align_valid_routes(ids, count, 32, experts)
            padded_rows = int(padded.item())  # Test-only observation, never production.
            positions = sorted_ids[:padded_rows]
            actual = positions[positions < capacity * topk]
            expected = (ids[:valid].reshape(-1) >= 0).nonzero().flatten().to(torch.int32)
            torch.testing.assert_close(actual.sort().values, expected, rtol=0, atol=0)
            if actual.numel():
                expert_for_row = expert_ids[:padded_rows // 32].repeat_interleave(32)
                assigned = expert_for_row[positions < capacity * topk]
                torch.testing.assert_close(ids.reshape(-1)[actual.long()].to(torch.int32), assigned, rtol=0, atol=0)
            kwargs = dict(w1=w1, w2=w2, topk_ids=ids, topk_weights=weights,
                          inplace=True, activation="silu", is_gated=True,
                          filter_expert=True)
            reference = fm.fused_experts_impl(hidden_states=x.clone(), **kwargs)
            candidate = fm.fused_experts_impl(hidden_states=x.clone(), valid_count=count, **kwargs)
            torch.cuda.synchronize()
            torch.testing.assert_close(candidate[:valid].view(torch.int16), reference[:valid].view(torch.int16), rtol=0, atol=0)
            results.append({"capacity": capacity, "valid": valid, "bitwise_equal": True,
                            "aligned_routes": actual.numel(), "padded_routes": padded_rows})
    print(json.dumps({"status": "passed", "alignment_cases": alignment_results,
                      "cases": results}, indent=2))


if __name__ == "__main__":
    main()
