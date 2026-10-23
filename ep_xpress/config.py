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
    fatal_errors: bool = False
    channels: int = 4  # allocated host_sm lanes; operations may activate a subset
    diagnostics: bool = False  # opt-in host_sm protocol diagnostics
    combine_frame_rows: int = 0  # 0: split chunk budget; positive: rows per pair/frame
    decode_max_tokens: int = 0  # 0: no decode mailbox; otherwise per-source capacity
    combine_warp_specialization: bool = False  # independent pair-queue send/receive groups

    def __post_init__(self):
        if not isinstance(self.combine_warp_specialization, bool):
            raise ValueError("combine_warp_specialization must be a bool")
        if self.combine_warp_specialization and self.transport != "host_sm":
            raise ValueError("combine_warp_specialization is only supported by host_sm")
        if not isinstance(self.fatal_errors, bool):
            raise ValueError("fatal_errors must be a bool")
        if self.fatal_errors and self.transport != "host_sm":
            raise ValueError("fatal_errors is only supported by host_sm")
        if not isinstance(self.diagnostics, bool):
            raise ValueError("diagnostics must be a bool")
        if self.diagnostics and self.transport != "host_sm":
            raise ValueError("diagnostics is only supported by host_sm")
        for name in ("hidden_size", "topk", "max_tokens_per_rank", "chunk_bytes",
                     "slots", "host_pool_budget_bytes", "max_pending_operations", "channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.channels > 32:
            raise ValueError("channels must be at most 32")
        for name, maximum in (("combine_frame_rows", 256), ("decode_max_tokens", 256)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [0, {maximum}]")
            if value and self.transport != "host_sm":
                raise ValueError(f"{name} is only supported by host_sm")
        if self.chunk_bytes < self.hidden_size * 2:
            raise ValueError("chunk_bytes must fit at least one BF16 row")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not math.isfinite(self.poll_interval_seconds) or self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be finite and nonnegative")
        if self.workload not in ("auto", "prefill", "decode"):
            raise ValueError("workload must be auto, prefill, or decode")
        if self.transport not in ("host_ce", "host_sm"):
            raise NotImplementedError("transport must be host_ce or host_sm")
        if self.transport == "host_sm":
            if self.hidden_size % 2 or self.hidden_size > (1 << 31) - 1:
                raise ValueError("host_sm requires an even hidden size that fits int32")
            if self.topk > 256 or self.slots > 64 or self.chunk_rows > 65536:
                raise ValueError("host_sm supports topk <= 256, slots <= 64 and chunk_rows <= 65536")
            if self.timeout_seconds > 3600 or self.max_tokens_per_rank > (1 << 31) - 1:
                raise ValueError("host_sm timeout must be <= 3600 seconds and token capacity must fit int32")
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
        if self.transport == "host_sm":
            return world_size * (self.mapped_arena_bytes(self.topk) +
                                 self.mapped_arena_bytes(0, queue_peers=world_size) +
                                 self.mailbox_arena_bytes(world_size))
        return 2 * world_size * self.slots * self.slot_bytes

    def channel_chunk_rows(self, channels=None):
        """Split the producer's payload budget across negotiated GPU lanes."""
        channels = self.channels if channels is None else channels
        return max(1, (self.chunk_rows + channels - 1) // channels)

    def combine_chunk_rows(self, channels=None, queue_peers=1):
        """Rows per pair/frame, independently configurable from channel count."""
        if (isinstance(queue_peers, bool) or not isinstance(queue_peers, int) or
                not 1 <= queue_peers <= 63):
            raise ValueError("queue_peers must be an integer in [1, 63]")
        if self.combine_frame_rows:
            return self.combine_frame_rows
        rows = self.channel_chunk_rows(channels)
        return max(1, (rows + queue_peers - 1) // queue_peers)

    def mailbox_arena_bytes(self, world_size):
        """One producer's decode mailbox, matching the native mapped layout."""
        if isinstance(world_size, bool) or not isinstance(world_size, int) or not 1 <= world_size <= 63:
            raise ValueError("world_size must be an integer in [1, 63]")
        if not self.decode_max_tokens:
            return 0
        token_bytes = ((32 + self.hidden_size * 2 + 127) // 128) * 128
        return ((128 + world_size * self.decode_max_tokens * token_bytes + 4095) // 4096) * 4096

    def mapped_arena_bytes(self, topk, channels=None, queue_peers=1):
        """Mapped queue allocation including per-row metadata and control words.

        Kept in agreement with the native ABI and checked during bootstrap.
        Dispatch uses one queue per channel. Combine uses one queue per peer
        and channel, splitting the payload budget before page/header rounding.
        Dispatch also reserves per-reader readiness and terminal-ACK words so
        non-readers do not gate payload-slot reuse. Active launch width never
        changes this initialized allocation or its frame size.
        """
        channels = self.channels if channels is None else channels
        if (isinstance(queue_peers, bool) or not isinstance(queue_peers, int) or
                not 1 <= queue_peers <= 63 or (topk and queue_peers != 1)):
            raise ValueError("queue_peers must be in [1, 63] and dispatch requires one queue")
        rows = (self.channel_chunk_rows(channels) if topk else
                self.combine_chunk_rows(channels, queue_peers))
        row_bytes = ((16 + topk * 12 + self.hidden_size * 2 + 15) // 16) * 16
        slot_bytes = ((640 + rows * row_bytes + 127) // 128) * 128
        notification_bytes = (self.slots * 63 + 64) * 8 if topk else 0
        return channels * queue_peers * (
            (128 + self.slots * slot_bytes + notification_bytes + 4095) // 4096) * 4096

    def pinned_working_set_bytes(self, world_size):
        """Shared rings plus all ranks' reusable private receive packs."""
        if self.transport == "host_sm":
            return self.pool_bytes(world_size)
        packs = world_size * (world_size - 1) * self.slots * self.chunk_rows * self.hidden_size * 2
        return self.pool_bytes(world_size) + packs
