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


def main():
    torch.manual_seed(173)
    device = "cuda"
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
        # Small single-launch alignment and large two-launch alignment; include
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
    print(json.dumps({"status": "passed", "cases": results}, indent=2))


if __name__ == "__main__":
    main()
