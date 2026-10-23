import unittest

from ep_xpress import Config


class ConfigTests(unittest.TestCase):
    def test_aligned_budget_and_rows(self):
        cfg = Config(10, 2, 128, chunk_bytes=100, slots=2)
        self.assertEqual(cfg.chunk_rows, 5)
        self.assertEqual(cfg.slot_bytes, 4096)
        self.assertEqual(cfg.pool_bytes(8), 2 * 8 * 2 * 4096)

    def test_reject_unsupported_and_invalid(self):
        for kw in ({"hidden_size": 0}, {"chunk_bytes": 1},
                   {"timeout_seconds": float("nan")}, {"slots": True}):
            args = dict(hidden_size=16, topk=2, max_tokens_per_rank=32)
            args.update(kw)
            with self.assertRaises(ValueError):
                Config(**args)
        for kw in ({"transport": "unknown"}, {"combine_backend": "cpu"},
                   {"graph_mode": "capture"}):
            with self.assertRaises(NotImplementedError):
                Config(16, 2, 32, **kw)

    def test_host_sm_mapped_budget_and_limits(self):
        cfg = Config(128, 4, 32, transport="host_sm", chunk_bytes=1024)
        self.assertEqual(cfg.chunk_rows, 4)
        self.assertEqual(cfg.pool_bytes(8), 8 * (cfg.mapped_arena_bytes(4) + cfg.mapped_arena_bytes(0, queue_peers=8)))
        self.assertEqual(cfg.pinned_working_set_bytes(8), cfg.pool_bytes(8))
        for changes in ({"hidden_size": 127}, {"topk": 257}, {"slots": 65},
                        {"chunk_bytes": 256 * 65537}, {"timeout_seconds": 3601},
                        {"fatal_errors": 1}, {"channels": 0}, {"channels": True},
                        {"channels": 33}):
            args = dict(hidden_size=128, topk=4, max_tokens_per_rank=32, transport="host_sm")
            args.update(changes)
            with self.assertRaises(ValueError):
                Config(**args)
        self.assertTrue(Config(128, 4, 32, transport="host_sm", fatal_errors=True).fatal_errors)
        with self.assertRaises(ValueError):
            Config(128, 4, 32, fatal_errors=True)

    def test_channel_budget_splits_payload_and_counts_lane_headers(self):
        cfg = Config(128, 4, 32, transport="host_sm", chunk_bytes=7 * 256)
        self.assertEqual(cfg.channels, 4)
        self.assertEqual(cfg.channel_chunk_rows(), 2)
        self.assertEqual(cfg.channel_chunk_rows(1), 7)
        self.assertEqual(cfg.channel_chunk_rows(3), 3)
        one_lane = Config(128, 4, 32, transport="host_sm", chunk_bytes=2 * 256, channels=1)
        self.assertEqual(cfg.mapped_arena_bytes(4), 4 * one_lane.mapped_arena_bytes(4))
        # The selective-reader READY and terminal-ACK suffix pushes this
        # dispatch lane across the next page boundary; combine has no suffix.
        self.assertEqual(cfg.mapped_arena_bytes(4), 4 * 8192)
        ce_one = Config(128, 4, 32, channels=1)
        ce_four = Config(128, 4, 32, channels=4)
        self.assertEqual(ce_one.pinned_working_set_bytes(8), ce_four.pinned_working_set_bytes(8))

    def test_pair_budget_uses_per_pair_rows_and_counts_all_headers(self):
        cfg = Config(128, 4, 32, transport="host_sm", chunk_bytes=73 * 256, channels=4)
        self.assertEqual(cfg.channel_chunk_rows(), 19)
        self.assertEqual(cfg.combine_chunk_rows(queue_peers=8), 3)
        row_stride = 272  # aligned BF16 payload plus row metadata
        slot_stride = ((640 + 3 * row_stride + 127) // 128) * 128
        lane_bytes = ((128 + 2 * slot_stride + 4095) // 4096) * 4096
        self.assertEqual(cfg.mapped_arena_bytes(0, queue_peers=8), 4 * 8 * lane_bytes)
        for peers in (0, 64, True, 1.5):
            with self.assertRaises(ValueError):
                cfg.mapped_arena_bytes(0, queue_peers=peers)
        with self.assertRaises(ValueError):
            cfg.mapped_arena_bytes(4, queue_peers=8)

    def test_host_sm_diagnostics_are_explicit_and_boolean(self):
        self.assertFalse(Config(128, 4, 32, transport="host_sm").diagnostics)
        self.assertTrue(Config(128, 4, 32, transport="host_sm", diagnostics=True).diagnostics)
        for value in (0, 1, None, "false"):
            with self.assertRaisesRegex(ValueError, "diagnostics must be a bool"):
                Config(128, 4, 32, transport="host_sm", diagnostics=value)
        with self.assertRaisesRegex(ValueError, "only supported by host_sm"):
            Config(128, 4, 32, transport="host_ce", diagnostics=True)
        self.assertFalse(Config(128, 4, 32, transport="host_ce").diagnostics)


if __name__ == "__main__":
    unittest.main()
