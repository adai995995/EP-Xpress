"""Host-CE data transport with bounded, independently progressing rings.

This reference implementation deliberately uses a CPU progress loop and CPU
gather into pinned receive buffers. It is eager-only, not GPU-initiated.
"""

import math
import time

import torch

from . import native


def exchange(context, tensor, destination_masks, receive_indices, operation, phase):
    """Export rows once per nonempty chunk; import only selected remote rows.

    receive_indices[source] is sorted in the source's row order. The caller
    owns all route validation. No peer device pointers are used here.
    """
    cfg = context.config
    rank = context.rank
    rings = context._rings[phase]
    own = rings[rank]
    rows_per_chunk = cfg.chunk_rows
    hidden = cfg.hidden_size
    row_bytes = hidden * 2
    outgoing_chunks = math.ceil(tensor.shape[0] / rows_per_chunk)
    # Source lengths include chunks not needed by this receiver. Only selected
    # chunks are queued, avoiding a dependency on unrelated consumers.
    imports = {}
    outputs = {}
    for source, indices in enumerate(receive_indices):
        if source == rank or not indices:
            continue
        outputs[source] = torch.empty((len(indices), hidden), dtype=torch.bfloat16,
                                      device=context.device)
        chunks = []
        offset = 0
        while offset < len(indices):
            chunk = indices[offset] // rows_per_chunk
            end = offset + 1
            while end < len(indices) and indices[end] // rows_per_chunk == chunk:
                end += 1
            relative = [index - chunk * rows_per_chunk for index in indices[offset:end]]
            chunks.append((chunk, offset, relative))
            offset = end
        imports[source] = chunks

    received_positions = {source: 0 for source in imports}
    pending_exports = {}
    pending_imports = []
    next_chunk = 0
    stats = {"d2h_bytes": 0, "h2d_bytes": 0, "cpu_pack_bytes": 0,
             "export_chunks": 0, "import_chunks": 0, "poll_iterations": 0,
             "backpressure_iterations": 0, "max_active_exports": 0}
    began = time.monotonic()
    deadline = began + cfg.timeout_seconds
    input_ready = torch.cuda.Event()
    input_ready.record(context._work_stream)
    context._export_stream.wait_event(input_ready)
    context._import_stream.wait_event(input_ready)
    tensor.record_stream(context._export_stream)

    while True:
        context._check_aborted()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{phase} timed out waiting for peer progress")
        progress = False
        stats["poll_iterations"] += 1

        # Publish only after D2H completion. Reclaim only after every actual
        # consumer has copied the immutable source rows into private storage.
        for chunk, (event, published) in list(pending_exports.items()):
            slot = chunk % cfg.slots
            sequence = (operation << 32) | chunk
            if not published and event.query():
                own.publish(slot, sequence)
                pending_exports[chunk] = (event, True)
                published = True
                progress = True
            if published and own.snapshot(slot)["state"] == "FREE":
                del pending_exports[chunk]
                progress = True

        # Always drain receives even when local outgoing slots have no credit.
        # This is essential when every producer has filled its ring.
        for source, chunks in imports.items():
            position = received_positions[source]
            if position == len(chunks):
                continue
            # Bound pinned receive buffers in flight by ring depth per source.
            if sum(item[2] == source for item in pending_imports) >= cfg.slots:
                continue
            pack_slot = position % cfg.slots
            if any(item[2] == source and item[3] == pack_slot for item in pending_imports):
                continue
            chunk, output_offset, relative = chunks[position]
            slot = chunk % cfg.slots
            sequence = (operation << 32) | chunk
            nrows = rings[source].ready(slot, sequence)
            if nrows is None:
                continue
            if max(relative) >= nrows:
                raise RuntimeError("published chunk does not contain planned rows")
            view = rings[source].payload_view(slot)
            host_rows = None
            try:
                host_rows = torch.frombuffer(view, dtype=torch.bfloat16,
                                             count=nrows * hidden).view(nrows, hidden)
                packed = context._pack_pool[source][pack_slot][:len(relative)]
                indices = torch.tensor(relative, dtype=torch.int64)
                torch.index_select(host_rows, 0, indices, out=packed)
            finally:
                del host_rows
                view.release()
            # CPU gather has completed. The source slot may now be reused;
            # the independent pinned pack stays alive until its H2D completes.
            rings[source].ack(slot, sequence, rank)
            with torch.cuda.stream(context._import_stream):
                destination = outputs[source][output_offset:output_offset + len(relative)]
                destination.copy_(packed, non_blocking=True)
                done = torch.cuda.Event()
                done.record(context._import_stream)
            outputs[source].record_stream(context._import_stream)
            pending_imports.append((done, packed, source, pack_slot))
            received_positions[source] += 1
            nbytes = len(relative) * row_bytes
            stats["h2d_bytes"] += nbytes
            stats["cpu_pack_bytes"] += nbytes
            stats["import_chunks"] += 1
            progress = True

        remaining = []
        for event, packed, source, pack_slot in pending_imports:
            if event.query():
                progress = True
            else:
                remaining.append((event, packed, source, pack_slot))
        pending_imports = remaining

        # Fill available slots, without waiting here for a busy one.
        while next_chunk < outgoing_chunks:
            start = next_chunk * rows_per_chunk
            count = min(rows_per_chunk, tensor.shape[0] - start)
            mask = 0
            for destinations in destination_masks[start:start + count]:
                mask |= destinations
            if mask == 0:
                next_chunk += 1
                progress = True
                continue
            slot = next_chunk % cfg.slots
            sequence = (operation << 32) | next_chunk
            if any(chunk % cfg.slots == slot for chunk in pending_exports):
                stats["backpressure_iterations"] += 1
                break
            if not own.try_claim(slot, sequence, mask, count):
                stats["backpressure_iterations"] += 1
                break
            with torch.cuda.stream(context._export_stream):
                native.copy_d2h_async(own.payload_address(slot), tensor, start,
                                      count, context._export_stream.cuda_stream)
                event = torch.cuda.Event()
                event.record(context._export_stream)
            pending_exports[next_chunk] = (event, False)
            next_chunk += 1
            stats["export_chunks"] += 1
            stats["d2h_bytes"] += count * row_bytes
            stats["max_active_exports"] = max(stats["max_active_exports"], len(pending_exports))
            progress = True

        receives_done = all(received_positions[source] == len(chunks)
                            for source, chunks in imports.items())
        if next_chunk == outgoing_chunks and not pending_exports and receives_done and not pending_imports:
            break
        if not progress:
            time.sleep(cfg.poll_interval_seconds)

    # Import events have completed on the host, but explicitly retain the
    # stream dependency for future implementations with deferred completion.
    imported = torch.cuda.Event()
    imported.record(context._import_stream)
    context._work_stream.wait_event(imported)
    stats["transport_seconds"] = time.monotonic() - began
    return outputs, stats
