import multiprocessing
import os
import stat
import struct
import tempfile
import time
import unittest
from pathlib import Path

from ep_xpress.ring import RingAbortedError, RingError, SharedRing


def _wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for shared ring progress")
        time.sleep(0.001)


def _controlled_reader(root, consumer, phase, connection):
    try:
        with SharedRing(Path(root), 0, phase, 2, 4096) as ring:
            _wait_until(lambda: ring.ready(0, 0) is not None)
            view = ring.payload_view(0)
            try:
                connection.send(("read", bytes(view[:7])))
            finally:
                view.release()
            if connection.recv() != "ack":
                raise AssertionError("unexpected reader command")
            ring.ack(0, 0, consumer)
            ring.ack(0, 0, consumer)
            connection.send(("acked", ring.snapshot(0)["state"]))
    finally:
        connection.close()


def _stream_reader(root, consumer, chunks, connection):
    try:
        with SharedRing(Path(root), 0, "dispatch", 2, 4096) as ring:
            for sequence in range(chunks):
                slot = sequence % 2
                _wait_until(lambda: ring.ready(slot, sequence) is not None)
                if consumer == 2:
                    time.sleep(0.002)
                view = ring.payload_view(slot)
                try:
                    if bytes(view[:64]) != struct.pack("<Q", sequence) * 8:
                        raise AssertionError("payload changed before its reader acknowledged it")
                finally:
                    view.release()
                ring.ack(slot, sequence, consumer)
            connection.send("complete")
    finally:
        connection.close()


def _abort_worker(root, connection):
    with SharedRing(Path(root), 0, "dispatch", 2, 4096) as ring:
        ring.abort()
        connection.send("aborted")
    connection.close()


class RingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.context = multiprocessing.get_context("spawn")
        self.processes = []
        self.connections = []
        self.rings = []

    def tearDown(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        for connection in self.connections:
            connection.close()
        for ring in reversed(self.rings):
            ring.close()
        self.directory.cleanup()

    def make_ring(self, phase="dispatch", create=True, **kwargs):
        ring = SharedRing(self.root, 0, phase, kwargs.get("slots", 2),
                          kwargs.get("slot_bytes", 4096), create=create)
        self.rings.append(ring)
        return ring

    def spawn(self, target, *args):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=target, args=(*args, child))
        process.start()
        child.close()
        self.processes.append(process)
        self.connections.append(parent)
        return process, parent

    def receive(self, connection):
        self.assertTrue(connection.poll(15), "worker did not report progress")
        return connection.recv()

    def finish(self, process):
        process.join(timeout=15)
        self.assertFalse(process.is_alive(), "worker failed to finish")
        self.assertEqual(process.exitcode, 0)

    def test_slow_reader_prevents_reuse_and_stale_ack_cannot_free_new_generation(self):
        ring = self.make_ring()
        self.assertTrue(ring.try_claim(0, 0, (1 << 1) | (1 << 2), 7))
        self.assertIsNone(ring.ready(0, 0))
        view = ring.payload_view(0)
        view[:7] = b"payload"
        view.release()
        ring.publish(0, 0)
        self.assertEqual(ring.ready(0, 0), 7)
        fast, fast_conn = self.spawn(_controlled_reader, self.root, 1, "dispatch")
        slow, slow_conn = self.spawn(_controlled_reader, self.root, 2, "dispatch")
        self.assertEqual(self.receive(fast_conn), ("read", b"payload"))
        self.assertEqual(self.receive(slow_conn), ("read", b"payload"))
        fast_conn.send("ack")
        self.assertEqual(self.receive(fast_conn), ("acked", "READY"))
        self.assertFalse(ring.try_claim(0, 2, 2, 1))
        self.assertEqual(ring.snapshot(0)["ack_mask"], 2)
        # A slow consumer blocks this slot, not the other slot.
        self.assertTrue(ring.try_claim(1, 1, 2, 1))
        ring.publish(1, 1)
        ring.ack(1, 1, 1)
        slow_conn.send("ack")
        self.assertEqual(self.receive(slow_conn), ("acked", "FREE"))
        self.finish(fast)
        self.finish(slow)
        ring.ack(0, 0, 1)  # Retry after all readers completed is still harmless.
        self.assertTrue(ring.try_claim(0, 2, 2, 1))
        with self.assertRaises(ValueError):
            ring.ack(0, 0, 1)
        self.assertEqual(ring.snapshot(0)["state"], "WRITING")
        ring.publish(0, 2)
        with self.assertRaises(ValueError):
            ring.ack(0, 0, 1)
        self.assertEqual(ring.snapshot(0)["state"], "READY")
        ring.ack(0, 2, 1)
        with self.assertRaises(ValueError):
            ring.try_claim(0, 2, 2, 1)

    def test_multiprocess_stream_reuses_both_slots(self):
        ring = self.make_ring()
        chunks = 32
        readers = [self.spawn(_stream_reader, self.root, rank, chunks) for rank in (1, 2)]
        for sequence in range(chunks):
            slot = sequence % 2
            _wait_until(lambda: ring.try_claim(slot, sequence, 6, 1))
            view = ring.payload_view(slot)
            view[:64] = struct.pack("<Q", sequence) * 8
            view.release()
            ring.publish(slot, sequence)
        for process, connection in readers:
            self.assertEqual(self.receive(connection), "complete")
            self.finish(process)
        self.assertEqual([ring.snapshot(i)["state"] for i in range(2)], ["FREE", "FREE"])

    def test_phases_are_independent_across_processes(self):
        dispatch = self.make_ring("dispatch")
        combine = self.make_ring("combine")
        for ring in (dispatch, combine):
            self.assertTrue(ring.try_claim(0, 0, 2, 1))
            view = ring.payload_view(0)
            view[:7] = b"payload"
            view.release()
            ring.publish(0, 0)
        process, connection = self.spawn(_controlled_reader, self.root, 1, "dispatch")
        self.assertEqual(self.receive(connection), ("read", b"payload"))
        connection.send("ack")
        self.assertEqual(self.receive(connection), ("acked", "FREE"))
        self.finish(process)
        self.assertEqual(dispatch.snapshot(0)["state"], "FREE")
        self.assertEqual(combine.snapshot(0)["state"], "READY")
        self.assertFalse(combine.try_claim(0, 2, 2, 1))
        combine.ack(0, 0, 1)

    def test_abort_is_visible_to_other_processes(self):
        ring = self.make_ring()
        process, connection = self.spawn(_abort_worker, self.root)
        self.assertEqual(self.receive(connection), "aborted")
        self.finish(process)
        self.assertTrue(ring.aborted())
        with self.assertRaises(RingAbortedError):
            ring.try_claim(0, 0, 2, 1)
        with self.assertRaises(RingAbortedError):
            ring.ready(0, 0)
        self.assertEqual(ring.snapshot(0)["state"], "FREE")

    def test_storage_validation_permissions_and_view_lifetime(self):
        ring = self.make_ring()
        for path in self.root.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(len(ring.data), 8192)
        self.assertEqual(ring.payload_address(1) - ring.payload_address(0), 4096)
        peer = self.make_ring(create=False)
        self.assertFalse(peer.aborted())
        with self.assertRaises(RingError):
            self.make_ring(create=True)
        with self.assertRaises(RingError):
            self.make_ring(create=False, slots=3)
        with self.assertRaises(ValueError):
            SharedRing(self.root, 0, "../invalid", 2, 4096)
        with self.assertRaises(ValueError):
            SharedRing(self.root, 0, "dispatch", 2, 4097)
        view = ring.payload_view(0)
        with self.assertRaises(BufferError):
            ring.close()
        view.release()
        ring.close()
        ring.close()
        with self.assertRaises(RingError):
            ring.snapshot(0)

    def test_invalid_protocol_operations(self):
        ring = self.make_ring()
        with self.assertRaises(ValueError):
            ring.try_claim(0, 0, 0, 1)
        self.assertTrue(ring.try_claim(0, 0, 2, 0))
        with self.assertRaises(ValueError):
            ring.publish(0, 1)
        with self.assertRaises(ValueError):
            ring.ack(0, 0, 1)
        ring.publish(0, 0)
        self.assertEqual(ring.ready(0, 0), 0)
        self.assertIsNone(ring.ready(0, 1))
        with self.assertRaises(ValueError):
            ring.ack(0, 0, 2)
        with self.assertRaises(ValueError):
            ring.publish(0, 0)
        ring.ack(0, 0, 1)
        self.assertIsNone(ring.ready(0, 0))


if __name__ == "__main__":
    unittest.main()
