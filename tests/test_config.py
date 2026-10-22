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
        self.assertEqual(cfg.pool_bytes(8), 8 * (cfg.mapped_arena_bytes(4) + cfg.mapped_arena_bytes(0)))
        self.assertEqual(cfg.pinned_working_set_bytes(8), cfg.pool_bytes(8))
        for changes in ({"hidden_size": 127}, {"topk": 257}, {"slots": 65},
                        {"chunk_bytes": 256 * 65537}, {"timeout_seconds": 3601},
                        {"fatal_errors": 1}):
            args = dict(hidden_size=128, topk=4, max_tokens_per_rank=32, transport="host_sm")
            args.update(changes)
            with self.assertRaises(ValueError):
                Config(**args)
        self.assertTrue(Config(128, 4, 32, transport="host_sm", fatal_errors=True).fatal_errors)
        with self.assertRaises(ValueError):
            Config(128, 4, 32, fatal_errors=True)


if __name__ == "__main__":
    unittest.main()
