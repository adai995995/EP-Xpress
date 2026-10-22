"""Validated resource limits; no machine-specific configuration."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    hidden_size: int
    topk: int
    max_tokens_per_rank: int
    chunk_bytes: int = 2 << 20
    slots: int = 2
    host_pool_budget_bytes: int = 512 << 20
    timeout_seconds: float = 60.0
    poll_interval_seconds: float = 0.0001
    max_pending_operations: int = 8
    workload: str = "auto"
    transport: str = "host_ce"
    combine_backend: str = "gpu"
    graph_mode: str = "eager"

    def __post_init__(self):
        for name in ("hidden_size", "topk", "max_tokens_per_rank", "chunk_bytes",
                     "slots", "host_pool_budget_bytes", "max_pending_operations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_bytes < self.hidden_size * 2:
            raise ValueError("chunk_bytes must fit at least one BF16 row")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not math.isfinite(self.poll_interval_seconds) or self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be finite and nonnegative")
        if self.workload not in ("auto", "prefill", "decode"):
            raise ValueError("workload must be auto, prefill, or decode")
        if self.transport != "host_ce":
            raise NotImplementedError("this release implements host_ce only")
        if self.combine_backend != "gpu":
            raise NotImplementedError("this release implements GPU combine only")
        if self.graph_mode != "eager":
            raise NotImplementedError("CUDA Graph capture is not implemented")

    @property
    def chunk_rows(self):
        return self.chunk_bytes // (self.hidden_size * 2)

    @property
    def slot_bytes(self):
        return ((self.chunk_rows * self.hidden_size * 2 + 4095) // 4096) * 4096

    def pool_bytes(self, world_size):
        return 2 * world_size * self.slots * self.slot_bytes

    def pinned_working_set_bytes(self, world_size):
        """Shared rings plus all ranks' reusable private receive packs."""
        packs = world_size * (world_size - 1) * self.slots * self.chunk_rows * self.hidden_size * 2
        return self.pool_bytes(world_size) + packs
