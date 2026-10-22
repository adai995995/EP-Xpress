import copy
import math
import random
import unittest

from ep_xpress.routing import build_routes


class RoutingTests(unittest.TestCase):
    def test_rank_dedup_inverse_mapping_and_unmodified_weights(self):
        ids = [[[0, 2, 1], [3, -1, -1]], [[1, 3, -1]], []]
        weights = [[[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]], [[0.7, 0.3, 0.0]], []]
        original = copy.deepcopy((ids, weights))
        plan = build_routes(ids, weights, [0, 1, 0, 2], 3, 3)
        self.assertEqual(plan.rows, [[(0, 0)], [(0, 0), (1, 0)], [(0, 1), (1, 0)]])
        self.assertEqual(plan.local_ids, [
            [[0, 1, -1]], [[-1, -1, 0], [0, -1, -1]],
            [[0, -1, -1], [-1, 0, -1]],
        ])
        self.assertEqual(plan.dispatch_dest_masks, [[2, 4], [4], []])
        self.assertEqual(plan.dispatch_recv_indices[2], [[1], [0], []])
        self.assertEqual(plan.combine_recv_indices[0], [[0], [0], [0]])
        self.assertEqual(plan.combine_token_indices[0], [[0], [0], [1]])
        self.assertEqual(plan.combine_recv_indices[1], [[], [1], [1]])
        self.assertEqual(plan.weights[0][0], weights[0][0])
        self.assertEqual(plan.weights[1][0], weights[0][0])
        self.assertEqual((ids, weights), original)
        weights[0][0][0] = 100
        self.assertEqual(plan.weights[0][0][0], 0.2)
        self.assertEqual(plan.weights[1][0][0], 0.2)

    def test_local_only_rank_and_empty_ranks(self):
        plan = build_routes([[], [[0, 1]], []], [[], [[0.4, 0.6]], []], [1, 1], 3, 2)
        self.assertEqual(plan.rows, [[], [(1, 0)], []])
        self.assertEqual(plan.dispatch_dest_masks, [[], [0], []])
        self.assertEqual(plan.combine_recv_indices[1][1], [0])
        self.assertEqual(plan.local_ids[1], [[0, 1]])
        empty = build_routes([[], []], [[], []], [0], 2, 1)
        self.assertEqual(empty.rows, [[], []])
        self.assertEqual(empty.token_counts, [0, 0])

    def test_high_rank_mask(self):
        ids = [[[0]]] + [[] for _ in range(62)]
        weights = [[[1.0]]] + [[] for _ in range(62)]
        plan = build_routes(ids, weights, [62], 63, 1)
        self.assertEqual(plan.dispatch_dest_masks[0], [1 << 62])

    def test_dropped_rows_preserve_source_positions(self):
        ids = [[[-1, -1], [0, 2], [-1, -1], [1, -1]], [[-1, -1]]]
        # Masked slots may retain router-produced weights. Their IDs, rather
        # than their weights, determine whether a contribution exists.
        weights = [[[0.3, 0.7], [0.4, 0.6], [0.8, 0.2], [1.0, 0.0]], [[0.5, 0.5]]]
        plan = build_routes(ids, weights, [0, 1, 1], 2, 2)
        self.assertEqual(plan.token_counts, [4, 1])
        self.assertEqual(plan.rows, [[(0, 1)], [(0, 1), (0, 3)]])
        self.assertEqual(plan.dispatch_dest_masks, [[0, 2, 0, 2], [0]])
        self.assertEqual(plan.dispatch_recv_indices, [[[1], []], [[1, 3], []]])
        self.assertEqual(plan.combine_token_indices, [[[1], [1, 3]], [[], []]])
        self.assertEqual(plan.combine_recv_indices, [[[0], [0, 1]], [[], []]])

    def test_all_dropped_rows_keep_counts_without_routes(self):
        plan = build_routes([[[-1, -1], [-1, -1]], [[-1, -1]], []],
                            [[[0.5, 0.5], [0.2, 0.8]], [[1.0, 0.0]], []],
                            [0, 1, 2], 3, 2)
        self.assertEqual(plan.token_counts, [2, 1, 0])
        self.assertEqual(plan.rows, [[], [], []])
        self.assertEqual(plan.local_ids, [[], [], []])
        self.assertEqual(plan.weights, [[], [], []])
        self.assertEqual(plan.dispatch_dest_masks, [[0, 0], [0], []])
        for indices in (plan.dispatch_recv_indices, plan.combine_recv_indices,
                        plan.combine_token_indices):
            self.assertEqual(indices, [[[], [], []], [[], [], []], [[], [], []]])

    def test_masked_nonfinite_weights_are_zeroed_without_changing_active_weights(self):
        ids = [[[0, -1, 1], [-1, -1, -1]], [[-1, 1, -1]]]
        weights = [[[0.4, math.nan, 0.6], [math.nan, math.inf, -math.inf]],
                   [[math.inf, 0.75, math.nan]]]
        plan = build_routes(ids, weights, [0, 1], 2, 3)
        self.assertEqual(plan.token_counts, [2, 1])
        self.assertEqual(plan.rows, [[(0, 0)], [(0, 0), (1, 0)]])
        self.assertEqual(plan.weights, [[[0.4, 0.0, 0.6]],
                                       [[0.4, 0.0, 0.6], [0.0, 0.75, 0.0]]])
        self.assertTrue(math.isnan(weights[0][0][1]))
        all_dropped = build_routes([[[-1, -1]]], [[[math.nan, math.inf]]], [0], 1, 2)
        self.assertEqual(all_dropped.rows, [[]])
        self.assertEqual(all_dropped.token_counts, [1])

    def test_active_nonfinite_weights_still_fail_with_route_coordinates(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, r"rank=1, token=1, slot=1, expert=0"):
                build_routes([[], [[-1, -1], [-1, 0]]],
                             [[], [[math.nan, math.inf], [math.nan, value]]],
                             [0], 2, 2)

    def test_invalid_shapes_ids_and_weights(self):
        valid = ([[[0, -1]]], [[[1.0, 0.0]]], [0], 1, 2)
        replacements = [
            (0, []), (1, []), (0, [[[0]]]), (1, [[[1.0]]]),
            (0, [[[0, 0]]]), (0, [[[1, -1]]]),
            (0, [[[-2, -1]]]), (0, [[[0.0, -1]]]), (0, [[[True, -1]]]),
            (1, [[[math.inf, 0.0]]]), (1, [[[math.nan, 0.0]]]),
            (1, [[[True, 0.0]]]), (1, [[['1.0', 0.0]]]),
            (2, []), (2, [1]), (3, 0), (3, 64), (3, True), (4, 0),
        ]
        for index, value in replacements:
            with self.subTest(index=index, value=value):
                args = list(valid)
                args[index] = value
                with self.assertRaises(ValueError):
                    build_routes(*args)

    def test_randomized_dispatch_compute_combine_matches_expert_reference(self):
        rng = random.Random(91)
        for _ in range(50):
            world, experts, topk = 4, 11, 4
            owners = [rng.randrange(world) for _ in range(experts)]
            ids, weights, hidden = [], [], []
            for source in range(world):
                count = rng.randrange(6)
                rank_ids, rank_weights, rank_hidden = [], [], []
                for token in range(count):
                    active = rng.sample(range(experts), rng.randrange(topk + 1))
                    rank_ids.append(active + [-1] * (topk - len(active)))
                    # Deliberately not normalized, to detect unwanted scaling.
                    rank_weights.append([rng.uniform(-1, 2) for _ in range(topk)])
                    rank_hidden.append([rng.uniform(-2, 2) for _ in range(3)])
                ids.append(rank_ids)
                weights.append(rank_weights)
                hidden.append(rank_hidden)
            plan = build_routes(ids, weights, owners, world, topk)
            reference = []
            for source in range(world):
                reference.append([
                    [sum(weights[source][t][k] * (expert + 1) * value
                         for k, expert in enumerate(ids[source][t]) if expert >= 0)
                     for value in hidden[source][t]]
                    for t in range(len(hidden[source]))
                ])
            partials = [[] for _ in range(world)]
            for receiver in range(world):
                global_ids = [expert for expert, owner in enumerate(owners) if owner == receiver]
                self.assertEqual(plan.rows[receiver], sorted(set(plan.rows[receiver])))
                for row, (source, token) in enumerate(plan.rows[receiver]):
                    partials[receiver].append([
                        sum(plan.weights[receiver][row][k] * (global_ids[local] + 1) * value
                            for k, local in enumerate(plan.local_ids[receiver][row]) if local >= 0)
                        for value in hidden[source][token]
                    ])
            output = [[[0.0] * 3 for _ in rank] for rank in hidden]
            for owner in range(world):
                for receiver in range(world):
                    for row, token in zip(plan.combine_recv_indices[owner][receiver],
                                          plan.combine_token_indices[owner][receiver]):
                        for dim, value in enumerate(partials[receiver][row]):
                            output[owner][token][dim] += value
                for actual, expected in zip(output[owner], reference[owner]):
                    for a, b in zip(actual, expected):
                        self.assertAlmostEqual(a, b, places=10)


if __name__ == "__main__":
    unittest.main()
