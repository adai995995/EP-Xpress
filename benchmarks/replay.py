"""Deterministic multi-GPU dispatch/combine correctness and timing replay.

Run with torchrun --standalone --nproc-per-node=N --module benchmarks.replay.
The synthetic expert function is not a model or an expert-GEMM benchmark.
"""

import argparse
from datetime import timedelta
import json
import os
import time

import torch
import torch.distributed as dist

from ep_xpress import Config, Context, native


def input_count(rank, world, tokens, case, iteration):
    count = tokens
    if case == "empty":
        count = 0 if rank == iteration % world else max(tokens - rank, 1)
    if case == "all_empty":
        count = 0
    return count


def inputs(rank, world, tokens, hidden, case, iteration, device, id_dtype=torch.int64):
    count = input_count(rank, world, tokens, case, iteration)
    x = ((torch.arange(count * hidden, device=device).reshape(count, hidden) + 7 * rank)
         .remainder(29).float() / 128).to(torch.bfloat16)
    # The background repeats after 29 rows. Exact BF16 source/token markers
    # prevent a misplaced payload from passing just because that pattern repeats.
    token_index = torch.arange(count, device=device)
    markers = (
        torch.full_like(token_index, rank),
        token_index.remainder(128),
        token_index.div(128, rounding_mode="floor").remainder(128),
        token_index.div(128 * 128, rounding_mode="floor").remainder(128),
        torch.full_like(token_index, iteration % 128),
    )
    for column, marker in enumerate(markers[:hidden]):
        x[:, column] = (marker.float() / 128).to(torch.bfloat16)
    rows = []
    for token in range(count):
        row = []
        for slot in range(4):
            owner = rank if case == "local" else (0 if case == "hotspot" else
                    (rank + token + slot // 2 + iteration) % world)
            row.append(owner * 4 + slot)
        if case == "all_padding" or (case == "padding" and
                (rank == iteration % world or (token + rank + iteration) % 3 == 0)):
            row = [-1] * 4
        elif case == "padding" and (token + rank) % 2 == 0:
            row[-1] = -1
        rows.append(row)
    ids = torch.tensor(rows, dtype=id_dtype, device=device).reshape(count, 4)
    weights = torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float32,
                           device=device).expand(count, 4).contiguous()
    if case in ("padding", "all_padding"):
        # CUDA framework routers mask IDs; ignored weights may remain garbage.
        weights = weights.masked_fill(ids == -1, float("nan"))
        weights[:, 1::2] = torch.where(ids[:, 1::2] == -1, float("inf"), weights[:, 1::2])
    return x, ids, weights


def expert_partial(batch, rank):
    present = batch.local_ids >= 0
    factors = (rank * 4 + batch.local_ids + 1).float() / 8
    coefficient = torch.where(present, factors * batch.weights, 0).sum(dim=1)
    return (batch.hidden.float() * coefficient[:, None]).to(torch.bfloat16).contiguous()


def reference(x, ids, weights, rank, world):
    remote = torch.zeros_like(x, dtype=torch.float32)
    local = torch.zeros_like(x)
    for expert_rank in range(world):
        coefficient = torch.where(ids // 4 == expert_rank,
                                  (ids + 1).float() / 8 * weights, 0).sum(dim=1)
        partial = (x.float() * coefficient[:, None]).to(torch.bfloat16)
        if expert_rank == rank:
            local = partial
        else:
            remote.add_(partial.float())
    return (remote.to(torch.bfloat16).float() + local.float()).to(torch.bfloat16)


def check_received(batch, rank, world, args, case, iteration, device):
    """Validate after synchronization; count reads are not transport operations."""
    count = (int(batch.valid_count.item()) if args.transport == "host_sm"
             else batch.valid_count)
    assert 0 <= count <= batch.hidden.shape[0]
    assert torch.isfinite(batch.weights).all().item()
    expected_hidden, expected_ids, expected_weights, expected_inverse = [], [], [], []
    for source in range(world):
        x, ids, weights = inputs(source, world, args.tokens, args.hidden, case, iteration, device)
        selected = (ids // 4 == rank).any(dim=1)
        tokens = selected.nonzero().flatten()
        selected_ids = ids[selected]
        local = selected_ids // 4 == rank
        expected_hidden.append(x[selected])
        expected_ids.append(torch.where(local, selected_ids % 4, -1))
        expected_weights.append(torch.where(local, weights[selected], 0))
        expected_inverse.append(torch.stack((torch.full_like(tokens, source), tokens), dim=1))
    expected_hidden = torch.cat(expected_hidden)
    expected_ids = torch.cat(expected_ids)
    expected_weights = torch.cat(expected_weights)
    assert count == expected_hidden.shape[0]
    hidden, ids, weights = batch.hidden[:count], batch.local_ids[:count], batch.weights[:count]
    if args.transport == "host_sm":
        inverse = batch.handle.inverse_map[:count]
        # Peers may arrive in any order. The inverse map must still uniquely
        # identify each source token exactly once per destination rank.
        keys = inverse[:, 0] * (args.tokens + 1) + inverse[:, 1]
        order = keys.argsort()
        torch.testing.assert_close(inverse[order], torch.cat(expected_inverse), rtol=0, atol=0)
        hidden, ids, weights = hidden[order], ids[order], weights[order]
        assert (batch.local_ids[count:] == -1).all().item()
        assert (batch.weights[count:] == 0).all().item()
        assert (batch.hidden[count:] == 0).all().item()
    else:
        assert count == batch.hidden.shape[0]
    torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(ids, expected_ids.to(ids.dtype), rtol=0, atol=0)
    torch.testing.assert_close(torch.where(ids >= 0, weights, 0), expected_weights, rtol=0, atol=0)


def check_expected_error(context, rank, world, args, device):
    """One deliberately failed operation; a failed context must be discarded."""
    x, ids, weights = inputs(rank, world, args.tokens, args.hidden, "mixed", 0, device,
                            id_dtype=getattr(torch, args.id_dtype))
    capacity = world * args.tokens
    chosen_batch = None
    if args.expect_error == "mismatch-handle":
        first = context.dispatch_async(x, ids, weights, tag="first", recv_capacity=capacity).wait()
        second = context.dispatch_async(x, ids, weights, tag="second", recv_capacity=capacity).wait()
        context.check_errors()
        # Preparatory operations must succeed everywhere before injecting the
        # mismatch, so an unrelated dispatch failure cannot satisfy this test.
        dist.barrier()
        chosen_batch = first if rank == 0 else second
    if rank == 0:
        if args.expect_error == "capacity":
            capacity = 0
        elif args.expect_error == "invalid-id":
            ids[0, 0] = world * 4
        elif args.expect_error == "nonfinite-weight":
            weights[0, 0] = float("nan")
    caught = False
    try:
        if chosen_batch is not None:
            context.combine_async(expert_partial(chosen_batch, rank), chosen_batch.handle,
                                  tag="same-combine-tag").wait()
        else:
            tag = "different-tag" if args.expect_error == "mismatch-tag" and rank == 0 else "same-tag"
            context.dispatch_async(x, ids, weights, tag=tag, recv_capacity=capacity).wait()
        context.check_errors()
    except RuntimeError as error:
        if not (str(error).startswith("host_sm:") or
                str(error).startswith("EP context is in a failed state")):
            raise
        if "timed out" in str(error) and args.expect_error.startswith("mismatch-"):
            raise AssertionError("mismatched operations must be rejected without timing out") from error
        caught = True
    flags = [None] * world
    dist.all_gather_object(flags, caught)
    if not all(flags):
        raise AssertionError(f"device failure was not observed on all ranks: {flags}")


def channel_configuration(context, world, args):
    effective = context.effective_channels if args.transport == "host_sm" else 1
    capacities = []
    if args.transport == "host_sm":
        capacities = [None] * world
        dist.all_gather_object(capacities, native.host_sm_max_channels())
    if args.require_channels is not None and effective != args.require_channels:
        raise AssertionError(
            f"test requires {args.require_channels} active channels, got {effective}; "
            f"cooperative capacities are {capacities}")
    return {"requested_channels": args.channels, "effective_channels": effective,
            "cooperative_channel_capacities": capacities}


def check_combine_profile(context, rank, world, args, cases):
    """Read diagnostics only after all captured operations have completed."""
    report = context.stop_combine_profile()
    assert report["status"] == 0 and not report["host_failed"]
    expected_operations = [(iteration, case) for iteration in range(args.iterations) for case in cases]
    assert report["captured_operations"] == len(expected_operations)
    assert report["dropped_operations"] == 0
    assert len(report["operations"]) == len(report["raw_counters"]) == len(expected_operations)
    phases = (
        "route_local_ns", "send_metadata_ns", "send_payload_ns", "publish_fence_ready_ns",
        "peer_poll_ns", "receive_metadata_select_ns", "receive_payload_ns",
        "ack_publish_ns", "other_control_ns",
    )
    waits = ("idle_wait_ready_ns", "idle_wait_credit_ns", "idle_wait_both_ns",
             "idle_final_drain_ns", "idle_other_ns")
    summaries = []
    for (iteration, case), operation, counters in zip(
            expected_operations, report["operations"], report["raw_counters"]):
        assert operation["submitted"]
        assert operation["tag"] == [iteration, case]
        assert operation["label"] == {"iteration": iteration, "case": case}
        assert operation["local_tokens"] == input_count(rank, world, args.tokens, case, iteration)
        assert len(counters) == context.effective_channels
        rows = [dict(zip(report["fields"], values)) for values in counters]
        for row in rows:
            assert row["status"] == 0
            assert row["end_ns"] - row["begin_ns"] == row["elapsed_ns"] > 0
            assert sum(row[field] for field in phases) == row["elapsed_ns"]
            assert all(row[field] >= 0 for field in phases + waits)
            assert sum(row[field] for field in waits) <= row["elapsed_ns"]
            assert row["idle_iterations"] + row["progress_iterations"] == row["loop_iterations"]
            assert row["credit_blocked_progress_iterations"] <= row["progress_iterations"]
            assert row["ready_missing_progress_iterations"] <= row["progress_iterations"]
            assert row["terminal_sent"] == 1
            assert row["terminal_received"] == world - 1
            assert row["send_frames"] >= row["terminal_sent"]
            assert row["receive_frames"] >= row["terminal_received"]
            assert row["payload_written_bytes"] == row["send_payload_rows"] * args.hidden * 2
            assert row["payload_read_bytes"] == row["receive_payload_rows"] * args.hidden * 2
            for field in ("first_publish_elapsed_ns", "first_receive_elapsed_ns",
                          "first_payload_publish_elapsed_ns", "first_payload_receive_elapsed_ns"):
                assert row[field] == -1 or 0 <= row[field] <= row["elapsed_ns"]
            assert (row["first_payload_publish_elapsed_ns"] == -1) == (row["send_payload_rows"] == 0)
            assert (row["first_payload_receive_elapsed_ns"] == -1) == (row["receive_payload_rows"] == 0)
        # Reconstruct the fixture's routing on CPU after timing, independently
        # of device counters and the received inverse map. Deduplicate owners.
        expected = {"local_rows": 0, "send_payload_rows": 0, "receive_payload_rows": 0}
        for source in range(world):
            _, ids, _ = inputs(source, world, args.tokens, 1, case, iteration, torch.device("cpu"))
            for ids_row in ids.tolist():
                owners = {expert // 4 for expert in ids_row if expert >= 0}
                if source == rank:
                    expected["local_rows"] += rank in owners
                    expected["receive_payload_rows"] += len(owners - {rank})
                else:
                    expected["send_payload_rows"] += rank in owners
        for field, value in expected.items():
            assert sum(row[field] for row in rows) == value, (case, iteration, field, value)
        summaries.append({
            "case": case, "iteration": iteration, **expected,
            "payload_written_bytes": sum(row["payload_written_bytes"] for row in rows),
            "payload_read_bytes": sum(row["payload_read_bytes"] for row in rows),
            "send_frames": sum(row["send_frames"] for row in rows),
            "receive_frames": sum(row["receive_frames"] for row in rows),
            "cta_elapsed_ns_min": min(row["elapsed_ns"] for row in rows),
            "cta_elapsed_ns_max": max(row["elapsed_ns"] for row in rows),
            "phase_partition_verified": True, "idle_subsets_verified": True,
        })
    return {"rank": rank, "channels": context.effective_channels,
            "captured_operations": report["captured_operations"], "operations": summaries}


def run(args):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout))
    config = Config(hidden_size=args.hidden, topk=4, max_tokens_per_rank=args.tokens,
                    chunk_bytes=args.chunk_bytes, slots=args.slots,
                    timeout_seconds=args.timeout, transport=args.transport, channels=args.channels)
    totals = []
    profile_summaries = None
    cases = args.cases.split(",")
    if args.expect_error:
        with Context(dist.group.WORLD, [owner for owner in range(world) for _ in range(4)],
                     config, device=device) as context:
            channel_info = channel_configuration(context, world, args)
            check_expected_error(context, rank, world, args, device)
        if rank == 0:
            print(json.dumps({"status": "passed", "transport": args.transport,
                              "expected_error": args.expect_error,
                              "routing_id_dtype": args.id_dtype, **channel_info}), flush=True)
        dist.destroy_process_group()
        return
    with Context(dist.group.WORLD, [owner for owner in range(world) for _ in range(4)],
                 config, device=device) as context:
        channel_info = channel_configuration(context, world, args)
        if args.profile_combine:
            context.start_combine_profile(max_operations=args.iterations * len(cases))
        for iteration in range(args.iterations):
            for case in cases:
                x, ids, weights = inputs(rank, world, args.tokens, args.hidden, case, iteration, device,
                                        id_dtype=getattr(torch, args.id_dtype))
                if args.profile_combine:
                    context.set_combine_profile_label({"iteration": iteration, "case": case})
                torch.cuda.synchronize()
                dist.barrier()
                start = time.perf_counter()
                if rank == args.delay_rank:
                    time.sleep(args.delay_ms / 1000)
                options = {}
                if args.transport == "host_sm":
                    options["recv_capacity"] = sum(
                        input_count(source, world, args.tokens, case, iteration)
                        for source in range(world))
                dispatch = context.dispatch_async(x, ids, weights, tag=[iteration, case], **options)
                batch = dispatch.wait()
                partial = expert_partial(batch, rank)
                if rank == args.delay_rank:
                    time.sleep(args.delay_ms / 1000)
                combine = context.combine_async(partial, batch.handle, tag=[iteration, case])
                output = combine.wait()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if args.transport == "host_sm":
                    context.check_errors()
                expected = reference(x, ids, weights, rank, world)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                assert output.shape == x.shape
                if case == "all_padding":
                    count = (batch.valid_count.item() if args.transport == "host_sm"
                             else batch.valid_count)
                    assert count == 0
                    if args.transport == "host_ce":
                        for metrics in (dispatch.metrics, combine.metrics):
                            assert metrics["d2h_bytes"] == metrics["h2d_bytes"] == 0
                check_received(batch, rank, world, args, case, iteration, device)
                record = {"case": case, "iteration": iteration,
                          "elapsed_ms": elapsed * 1000,
                          "dispatch": dispatch.metrics, "combine": combine.metrics}
                gathered = [None] * world
                dist.all_gather_object(gathered, record)
                if rank == 0:
                    result = {"case": case, "iteration": iteration,
                              "max_rank_ms": max(item["elapsed_ms"] for item in gathered)}
                    if args.transport == "host_ce":
                        for key in ("d2h_bytes", "h2d_bytes"):
                            result[key] = sum(item[phase][key] for item in gathered
                                              for phase in ("dispatch", "combine"))
                        result["max_active_exports"] = max(
                            item[phase]["max_active_exports"] for item in gathered
                            for phase in ("dispatch", "combine"))
                    else:
                        result["byte_metrics_available"] = False
                    totals.append(result)
                if args.check_handles and args.transport == "host_ce":
                    try:
                        context.combine_async(partial, batch.handle)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("consumed handle was accepted")

        if args.profile_combine:
            summary = check_combine_profile(context, rank, world, args, cases)
            profile_summaries = [None] * world
            dist.all_gather_object(profile_summaries, summary)
            for index in range(args.iterations * len(cases)):
                operations = [item["operations"][index] for item in profile_summaries]
                assert sum(item["payload_written_bytes"] for item in operations) == sum(
                    item["payload_read_bytes"] for item in operations)

        # Keep two dispatch handles alive and combine them in reverse order.
        if args.check_handles:
            first_x, first_ids, first_weights = inputs(
                rank, world, args.tokens, args.hidden, "mixed", 0, device,
                id_dtype=getattr(torch, args.id_dtype))
            second_x, second_ids, second_weights = inputs(
                rank, world, max(0, args.tokens - 1), args.hidden, "mixed", 1, device,
                id_dtype=getattr(torch, args.id_dtype))
            second_weights = second_weights * 0.5
            first = context.dispatch_async(first_x, first_ids, first_weights, tag="first")
            second = context.dispatch_async(second_x, second_ids, second_weights, tag="second")
            first_batch, second_batch = first.wait(), second.wait()
            # Distinct shapes, routes, weights and payloads expose accidental
            # reuse of the later dispatch's inverse map or valid-count storage.
            for batch, x, ids, weights in (
                (second_batch, second_x, second_ids, second_weights),
                (first_batch, first_x, first_ids, first_weights),
            ):
                output = context.combine_async(expert_partial(batch, rank), batch.handle).wait()
                if args.transport == "host_sm":
                    context.check_errors()
                assert output.shape == x.shape
                torch.testing.assert_close(output, reference(x, ids, weights, rank, world), rtol=0, atol=0)
            if args.transport == "host_sm":
                # host_sm propagates submission failures to peer GPUs; reject
                # handle reuse only after all successful work has been checked.
                dist.barrier()
                try:
                    context.combine_async(expert_partial(first_batch, rank), first_batch.handle)
                except ValueError:
                    pass
                else:
                    raise AssertionError("consumed handle was accepted")

    if rank == 0:
        print(json.dumps({"status": "passed", "kind": "synthetic_correctness_replay",
                          "world_size": world, "hidden_size": args.hidden,
                          "tokens_per_rank": args.tokens, "slots": args.slots,
                          "chunk_bytes": args.chunk_bytes, "graph": "eager",
                          "precision": "bf16", "transport": args.transport,
                          **channel_info,
                          "routing_id_dtype": args.id_dtype,
                          "artificial_delay_rank": args.delay_rank,
                          "artificial_delay_ms_per_phase": args.delay_ms,
                          "combine_diagnostics_enabled": args.profile_combine,
                          "combine_diagnostics": profile_summaries,
                          "timing_caveat": ("Instrumented diagnostic replay; CTA barriers and timers add overhead. "
                                            "Wait subsets overlap phase timings; device clocks are not compared."
                                            if args.profile_combine else None),
                          "results": totals}, indent=2), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--chunk-bytes", type=int, default=1024)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--cases", default="mixed,empty,local,hotspot,all_empty,padding,all_padding")
    parser.add_argument("--check-handles", action="store_true")
    parser.add_argument("--transport", choices=("host_ce", "host_sm"), default="host_ce")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--require-channels", type=int,
                        help="Fail if bootstrap falls back to a different active channel count")
    parser.add_argument("--id-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument("--profile-combine", action="store_true",
                        help="Capture bounded combine diagnostics and validate payload accounting after replay")
    parser.add_argument("--expect-error", choices=(
        "capacity", "invalid-id", "nonfinite-weight", "mismatch-tag", "mismatch-handle"))
    parser.add_argument("--delay-rank", type=int, default=-1,
                        help="Delay this rank before each phase to test queue progress")
    parser.add_argument("--delay-ms", type=float, default=0)
    args = parser.parse_args()
    unknown = set(args.cases.split(",")) - {
        "mixed", "empty", "local", "hotspot", "all_empty", "padding", "all_padding",
    }
    if unknown:
        parser.error("unknown replay case")
    if args.expect_error and (args.transport != "host_sm" or args.tokens < 1):
        parser.error("--expect-error requires host_sm and at least one input token")
    if args.profile_combine and (args.transport != "host_sm" or args.expect_error or args.iterations < 1):
        parser.error("--profile-combine requires a successful host_sm replay with positive iterations")
    if args.channels < 1 or (args.transport != "host_sm" and args.channels != 1):
        parser.error("--channels must be positive; multiple channels require host_sm")
    if args.require_channels is not None and not 1 <= args.require_channels <= args.channels:
        parser.error("--require-channels must be positive and no greater than --channels")
    if args.expect_error in ("mismatch-tag", "mismatch-handle") and int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("mismatched operations require at least two EP ranks")
    if args.delay_ms < 0 or args.delay_ms >= args.timeout * 1000:
        parser.error("--delay-ms must be nonnegative and below the operation timeout")
    if args.delay_rank < -1 or args.delay_rank >= int(os.environ.get("WORLD_SIZE", "1")):
        parser.error("--delay-rank must be -1 or a rank in the EP group")
    run(args)


if __name__ == "__main__":
    main()
