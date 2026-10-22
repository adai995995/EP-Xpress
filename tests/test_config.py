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
        for kw in ({"transport": "host_sm"}, {"combine_backend": "cpu"},
                   {"graph_mode": "capture"}):
            with self.assertRaises(NotImplementedError):
                Config(16, 2, 32, **kw)


if __name__ == "__main__":
    unittest.main()
