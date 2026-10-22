"""Stable, exact token-to-rank routing with no tensor-library dependency.

This is the control-plane reference implementation. It preserves active router
weights, merges repeated destination ranks, and records the inverse mapping
needed to return one weighted partial per destination rank and token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class RoutePlan:
    world_size: int
    topk: int
    expert_to_rank: List[int]
    token_counts: List[int]
    rows: List[List[Tuple[int, int]]]
    local_ids: List[List[List[int]]]
    weights: List[List[List[float]]]
    dispatch_recv_indices: List[List[List[int]]]
    dispatch_dest_masks: List[List[int]]
    combine_recv_indices: List[List[List[int]]]
    combine_token_indices: List[List[List[int]]]


def _integer(value, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} is outside the supported range")
    return int(value)


def _length(value, name: str) -> int:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return len(value)


def build_routes(
    ids_by_rank,
    weights_by_rank,
    expert_to_rank,
    world_size: int,
    topk: int,
) -> RoutePlan:
    """Build a deterministic dispatch and inverse-combine plan.

    ``-1`` denotes a padded expert slot. A row containing only ``-1`` is a
    dropped/padding token: it retains its source position, has no destinations,
    and has no combine contributions. Weights of ``-1`` slots are ignored and
    normalized to zero; only active slots require finite weights. An active
    expert cannot appear twice for the same token. Local expert IDs follow their order in
    ``expert_to_rank``. Empty source ranks and ranks owning no experts are
    supported. Local transfers remain in the plan; only
    ``dispatch_dest_masks`` excludes the source's own rank.
    """
    world_size = _integer(world_size, "world_size", 1, 63)
    topk = _integer(topk, "topk", 1, (1 << 31) - 1)
    if _length(ids_by_rank, "ids_by_rank") != world_size:
        raise ValueError("ids_by_rank must contain one entry per rank")
    if _length(weights_by_rank, "weights_by_rank") != world_size:
        raise ValueError("weights_by_rank must contain one entry per rank")
    num_experts = _length(expert_to_rank, "expert_to_rank")
    if not num_experts:
        raise ValueError("expert_to_rank must contain at least one expert")
    owners = [
        _integer(owner, "expert owner", 0, world_size - 1)
        for owner in expert_to_rank
    ]
    next_local = [0] * world_size
    expert_local_ids = []
    for owner in owners:
        expert_local_ids.append(next_local[owner])
        next_local[owner] += 1

    # Validate the entire input before constructing any externally visible plan.
    clean_ids = []
    clean_weights = []
    token_counts = []
    for source in range(world_size):
        count = _length(ids_by_rank[source], "rank expert IDs")
        if _length(weights_by_rank[source], "rank weights") != count:
            raise ValueError("expert IDs and weights must have equal token counts")
        token_counts.append(count)
        rank_ids, rank_weights = [], []
        for token in range(count):
            ids = ids_by_rank[source][token]
            weights = weights_by_rank[source][token]
            if _length(ids, "token expert IDs") != topk:
                raise ValueError("each expert ID row must have topk entries")
            if _length(weights, "token weights") != topk:
                raise ValueError("each weight row must have topk entries")
            ids = [
                _integer(expert, "expert ID", -1, num_experts - 1)
                for expert in ids
            ]
            active = [expert for expert in ids if expert != -1]
            if len(set(active)) != len(active):
                raise ValueError("a token cannot select the same expert twice")
            token_weights = []
            for slot, (expert, weight) in enumerate(zip(ids, weights)):
                if expert == -1:
                    # CUDA routers may mask the ID without initializing or
                    # clearing the corresponding padded weight.
                    token_weights.append(0.0)
                    continue
                try:
                    valid = (
                        not isinstance(weight, bool)
                        and isinstance(weight, Real)
                        and math.isfinite(weight)
                    )
                except (OverflowError, TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError(
                        "active router weights must be finite real numbers "
                        f"(rank={source}, token={token}, slot={slot}, expert={expert})"
                    )
                token_weights.append(weight)
            rank_ids.append(ids)
            rank_weights.append(token_weights)
        clean_ids.append(rank_ids)
        clean_weights.append(rank_weights)

    rows = [[] for _ in range(world_size)]
    local_ids = [[] for _ in range(world_size)]
    weights = [[] for _ in range(world_size)]
    dispatch_recv_indices = [[[] for _ in range(world_size)] for _ in range(world_size)]
    dispatch_dest_masks = [[] for _ in range(world_size)]
    combine_recv_indices = [[[] for _ in range(world_size)] for _ in range(world_size)]
    combine_token_indices = [[[] for _ in range(world_size)] for _ in range(world_size)]

    for source, rank_ids in enumerate(clean_ids):
        for token, ids in enumerate(rank_ids):
            destinations = sorted({owners[expert] for expert in ids if expert >= 0})
            mask = 0
            for destination in destinations:
                if destination != source:
                    mask |= 1 << destination
                partial_row = len(rows[destination])
                rows[destination].append((source, token))
                local_ids[destination].append([
                    expert_local_ids[expert]
                    if expert >= 0 and owners[expert] == destination else -1
                    for expert in ids
                ])
                weights[destination].append(list(clean_weights[source][token]))
                dispatch_recv_indices[destination][source].append(token)
                combine_recv_indices[source][destination].append(partial_row)
                combine_token_indices[source][destination].append(token)
            dispatch_dest_masks[source].append(mask)

    return RoutePlan(
        world_size, topk, list(owners), token_counts, rows, local_ids, weights,
        dispatch_recv_indices, dispatch_dest_masks, combine_recv_indices,
        combine_token_indices,
    )
