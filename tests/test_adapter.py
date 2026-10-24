"""Adapter contract tests runnable without CUDA, PyTorch, or SGLang."""

import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from collections import namedtuple

import ep_xpress
from ep_xpress.adapters.sglang_config import (
    contiguous_expert_map,
    select_active_channels,
    receive_capacity_from_scheduler,
    validate_runner_config,
    validate_server_args,
)


ROOT = Path(__file__).resolve().parents[1]


def runner_config(**changes):
    values = dict(
        hidden_size=16, top_k=2, num_experts=4, num_local_experts=2,
        params_dtype="bf16", num_fused_shared_experts=0, no_combine=False,
        apply_router_weight_on_input=False, use_tp_all_gather_activation=False,
        layer_id=3,
    )
    values.update(changes)
    return SimpleNamespace(**values)


class Tensor:
    def test_single_frame_auto_width_uses_bounds_and_preserves_overrides(self):
        common = dict(is_extend=False, allocated=32, adaptive=True,
                      decode_source_bound=64, dispatch_frame_rows=32)
        self.assertEqual(select_active_channels(1, **common), 2)
        self.assertEqual(select_active_channels(32, **common), 4)
        self.assertEqual(select_active_channels(1, decode_override=1, **common), 1)
        self.assertEqual(select_active_channels(1, **dict(common, dispatch_frame_rows=1)), 1)
        self.assertEqual(select_active_channels(1, **dict(common, is_extend=True)), 1)
        self.assertEqual(select_active_channels(1, **dict(common, adaptive=False)), 32)

    def __init__(self, shape):
        self.shape = shape

    def new_empty(self, shape, dtype=None):
        return Tensor(shape)


@contextmanager
def fake_adapter(rows=3, scheduler_counts=(2, 1), is_extend=False):
    calls = []
    topk_type = namedtuple("StandardTopKOutput", "topk_weights topk_ids router_logits")
    output_type = namedtuple("StandardDispatchOutput", "hidden_states hidden_states_scale topk_output")
    combine_type = namedtuple("StandardCombineInput", "hidden_states")
    batch = SimpleNamespace(
        hidden=Tensor((rows, 16)), weights=object(), local_ids=object(),
        handle=object(), valid_count=rows,
    )

    class Pending:
        def __init__(self, result):
            self.result = result

        def wait(self, stream=None):
            return self.result

    class Context:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.effective_channels = kwargs["config"].channels

        def create_workspace(self, **kwargs):
            workspace = SimpleNamespace(**kwargs)
            calls.append(("workspace", workspace))
            return workspace

        def dispatch_async(self, *args, **kwargs):
            calls.append(("dispatch", args, kwargs))
            return Pending(batch)

        def combine_async(self, *args, **kwargs):
            if args[0].shape != batch.hidden.shape:
                raise ValueError("partial shape does not match the dispatch handle")
            calls.append(("combine", args, kwargs))
            return Pending(Tensor((2, 16)))

        def close(self):
            calls.append(("close",))

    class BaseDispatcher:
        def __init__(self):
            pass

    modules = {}

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        modules[name] = value
        return value

    dist = module("torch.distributed", get_backend=lambda group: "gloo",
                  get_world_size=lambda group: 2, get_rank=lambda group: 0)
    cuda = SimpleNamespace(
        current_device=lambda: 0, current_stream=lambda device=None: "stream",
        is_current_stream_capturing=lambda: False,
    )
    module("torch", bfloat16="bf16", float32="fp32", cuda=cuda,
           distributed=dist, device=lambda kind, index: (kind, index))
    for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.moe",
                 "sglang.srt.layers.moe.token_dispatcher"):
        module(name)
    module("sglang.srt.layers.moe.token_dispatcher.base", BaseDispatcher=BaseDispatcher,
           DispatchOutputFormat=SimpleNamespace(STANDARD="standard"))
    module("sglang.srt.layers.moe.token_dispatcher.standard",
           StandardCombineInput=combine_type, StandardDispatchOutput=output_type)
    module("sglang.srt.layers.moe.topk", StandardTopKOutput=topk_type,
           TopKOutputChecker=SimpleNamespace(format_is_standard=lambda value: isinstance(value, topk_type)))
    module("sglang.srt.layers.dp_attention",
           get_global_dp_buffer_len=lambda: None if scheduler_counts is None else sum(scheduler_counts),
           get_is_extend_in_batch=lambda: is_extend)
    parallel = SimpleNamespace(tp_size=2, attn_dp_size=2, moe_ep_size=2, moe_ep_rank=0,
                               moe_ep_group=SimpleNamespace(cpu_group=object()))
    module("sglang.srt.runtime_context", get_parallel=lambda: parallel)
    name = "_ep_xpress_sglang_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "ep_xpress/adapters/sglang.py")
    adapter = importlib.util.module_from_spec(spec)
    modules[name] = adapter
    with patch.dict(sys.modules, modules), patch.dict(ep_xpress.__dict__, {"Context": Context}):
        spec.loader.exec_module(adapter)
        yield SimpleNamespace(module=adapter, calls=calls, batch=batch, cuda=cuda,
                              topk=topk_type, combine=combine_type)


class AdapterValidationTest(unittest.TestCase):
    def test_global_channel_policy(self):
        for capacity, extend, expected in ((0, False, 1), (1, False, 1), (32, False, 4),
                                           (257, False, 32), (3, True, 3), (512, True, 32)):
            self.assertEqual(select_active_channels(capacity, is_extend=extend,
                allocated=32, adaptive=True), expected)
        self.assertEqual(select_active_channels(1, is_extend=False,
            allocated=32, adaptive=False), 32)
        self.assertEqual(select_active_channels(1, is_extend=False,
            allocated=16, adaptive=True, decode_override=32), 16)

    def test_scheduler_capacity_bound(self):
        self.assertEqual(receive_capacity_from_scheduler(
            [3, 0], world_size=2, max_tokens_per_rank=4), 3)
        self.assertEqual(receive_capacity_from_scheduler(
            [0, 0], world_size=2, max_tokens_per_rank=4), 0)
        for counts in (None, [3], [5, 0], [-1, 2], [True, 2], [1.0, 2]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                receive_capacity_from_scheduler(counts, world_size=2, max_tokens_per_rank=4)

    def test_contiguous_mapping(self):
        self.assertEqual(contiguous_expert_map(6, 3), [0, 0, 1, 1, 2, 2])
        with self.assertRaises(ValueError):
            contiguous_expert_map(5, 3)

    def test_weighting_and_partition_guards(self):
        validate_runner_config(runner_config(), world_size=2, bf16="bf16")
        for changes in ({"no_combine": True}, {"apply_router_weight_on_input": True},
                        {"num_experts": 5}, {"num_fused_shared_experts": 1},
                        {"params_dtype": "fp16"}, {"use_tp_all_gather_activation": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_runner_config(runner_config(**changes), world_size=2, bf16="bf16")

    def test_server_scope_guards(self):
        model = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
        values = dict(tp_size=2, ep_size=2, dp_size=2, enable_dp_attention=True,
                      moe_runner_backend="triton", dtype="bfloat16")
        validate_server_args(SimpleNamespace(**values), model)
        for changes in ({"dp_size": 1}, {"nnodes": 2}, {"enable_eplb": True},
                        {"enable_two_batch_overlap": True}, {"enable_single_batch_overlap": True},
                        {"speculative_algorithm": "EAGLE"}, {"quantization": "fp8"},
                        {"dwdp_size": 2}, {"ep_join_mode": "scale"}, {"enable_pdmux": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_server_args(SimpleNamespace(**(values | changes)), model)


class AdapterContractTest(unittest.TestCase):
    def test_decode_workspace_reuse_and_prefill_fallback(self):
        with patch.dict(os.environ, {"EP_XPRESS_TRANSPORT": "host_sm"}):
            with fake_adapter(scheduler_counts=(2, 1)) as fixture:
                dispatcher = fixture.module.HostEPDispatcher(runner_config())
                topk = fixture.topk(object(), object(), object())
                dispatcher.dispatch(Tensor((2, 16)), topk)
                first = fixture.calls[-1][2]
                self.assertTrue(first["fused_combine"])
                workspace = first["workspace"]
                dispatcher.combine(fixture.combine(Tensor((3, 16))))
                dispatcher.dispatch(Tensor((2, 16)), topk)
                self.assertIs(fixture.calls[-1][2]["workspace"], workspace)
                dispatcher.combine(fixture.combine(Tensor((3, 16))))
                fixture.module.get_is_extend_in_batch = lambda: True
                dispatcher.dispatch(Tensor((2, 16)), topk)
                self.assertFalse(fixture.calls[-1][2]["fused_combine"])
                self.assertNotIn("workspace", fixture.calls[-1][2])
                self.assertEqual(sum(c[0] == "workspace" for c in fixture.calls), 1)

    def test_host_sm_keeps_device_count_off_host_path(self):
        class DeviceCount:
            def __bool__(self):
                raise AssertionError("device count must not be converted to bool")

            def __eq__(self, other):
                raise AssertionError("device count must not be inspected")

            def __ne__(self, other):
                raise AssertionError("device count must not be inspected")

        with patch.dict(os.environ, {"EP_XPRESS_TRANSPORT": "host_sm", "EP_XPRESS_CHANNELS": "3"}):
            with fake_adapter(rows=7, scheduler_counts=[2, 5]) as fixture:
                fixture.batch.valid_count = DeviceCount()
                dispatcher = fixture.module.HostEPDispatcher(runner_config())
                output = dispatcher.dispatch(Tensor((2, 16)), fixture.topk(object(), object(), object()))
                self.assertTrue(fixture.calls[0][1]["config"].fatal_errors)
                self.assertEqual(fixture.calls[0][1]["config"].channels, 3)
                self.assertEqual(fixture.calls[-1][2]["recv_capacity"], 7)
                self.assertEqual(output.topk_output.router_logits.shape, (7, 0))
                self.assertEqual(output.hidden_states.shape, (7, 16))
                self.assertIs(output.valid_count, fixture.batch.valid_count)
                self.assertEqual(fixture.calls[-1][2]["active_channels"], 1)
                dispatcher.combine(fixture.combine(Tensor((7, 16))))
                self.assertIs(fixture.calls[-1][1][1], fixture.batch.handle)

    def test_host_sm_reuses_scheduler_capacity(self):
        with patch.dict(os.environ, {"EP_XPRESS_TRANSPORT": "host_sm"}):
            with fake_adapter(scheduler_counts=[2, 5]) as fixture:
                dispatcher = fixture.module.HostEPDispatcher(runner_config())
                scheduler = sys.modules["sglang.srt.layers.dp_attention"]
                scheduler.get_global_dp_buffer_len = lambda: 11
                # The adapter imported the getter; replace that binding to
                # model the scheduler's next forward without inspecting routes.
                fixture.module.get_global_dp_buffer_len = scheduler.get_global_dp_buffer_len
                dispatcher.dispatch(Tensor((2, 16)), fixture.topk(object(), object(), object()))
                self.assertEqual(fixture.calls[-1][2]["recv_capacity"], 11)
                self.assertIsNone(fixture.calls[-1][2]["tag"])

    def test_no_second_id_mapping_or_weight_application(self):
        with fake_adapter() as fixture:
            dispatcher = fixture.module.HostEPDispatcher(runner_config())
            topk = fixture.topk(object(), object(), object())
            output = dispatcher.dispatch(Tensor((2, 16)), topk)
            self.assertIs(output.topk_output.topk_ids, fixture.batch.local_ids)
            self.assertIs(output.topk_output.topk_weights, fixture.batch.weights)
            dispatch_call = fixture.calls[-1]
            self.assertIs(dispatch_call[1][1], topk.topk_ids)
            self.assertIs(dispatch_call[1][2], topk.topk_weights)
            result = dispatcher.combine(fixture.combine(Tensor((3, 16))))
            self.assertEqual(result.shape, (2, 16))
            combine_call = fixture.calls[-1]
            self.assertIs(combine_call[1][1], fixture.batch.handle)
            self.assertTrue(combine_call[2]["weights_applied"])
            self.assertEqual(combine_call[2]["tag"], dispatch_call[2]["tag"])

    def test_reject_overwrite_and_combine_twice(self):
        with fake_adapter() as fixture:
            dispatcher = fixture.module.HostEPDispatcher(runner_config())
            topk = fixture.topk(object(), object(), object())
            dispatcher.dispatch(Tensor((2, 16)), topk)
            with self.assertRaises(RuntimeError):
                dispatcher.dispatch(Tensor((2, 16)), topk)
            dispatcher.combine(fixture.combine(Tensor((3, 16))))
            with self.assertRaises(RuntimeError):
                dispatcher.combine(fixture.combine(Tensor((3, 16))))
        with fake_adapter() as fixture:
            dispatcher = fixture.module.HostEPDispatcher(runner_config())
            dispatcher.dispatch(Tensor((2, 16)), fixture.topk(object(), object(), object()))
            # Shape validation belongs to the core. Its failure makes this
            # adapter unusable; it must not retry a partially submitted epoch.
            with self.assertRaises(ValueError):
                dispatcher.combine(fixture.combine(Tensor((3, 2, 16))))
            with self.assertRaises(RuntimeError):
                dispatcher.combine(fixture.combine(Tensor((3, 16))))

    def test_shared_context_and_zero_receive_rank(self):
        with fake_adapter(rows=0) as fixture:
            for layer in (3, 4):
                dispatcher = fixture.module.HostEPDispatcher(runner_config(layer_id=layer))
                dispatcher.dispatch(Tensor((0, 16)), fixture.topk(object(), object(), object()))
                dispatcher.combine(fixture.combine(Tensor((0, 16))))
            self.assertEqual(sum(call[0] == "init" for call in fixture.calls), 1)
            self.assertEqual(sum(call[0] == "combine" for call in fixture.calls), 2)
            fixture.module.close_contexts()
            self.assertEqual(fixture.calls[-1], ("close",))
            self.assertEqual(fixture.module._contexts, {})

    def test_graph_capture_rejected(self):
        with fake_adapter() as fixture:
            dispatcher = fixture.module.HostEPDispatcher(runner_config())
            fixture.cuda.is_current_stream_capturing = lambda: True
            with self.assertRaises(RuntimeError):
                dispatcher.dispatch(Tensor((2, 16)), fixture.topk(object(), object(), object()))
            self.assertEqual(fixture.calls, [])


@unittest.skipUnless(os.environ.get("EP_XPRESS_SGLANG_SOURCE"), "Set EP_XPRESS_SGLANG_SOURCE to audit patch application")
class SGLangPatchTest(unittest.TestCase):
    def test_patch_applies_and_modified_sources_parse(self):
        source = Path(os.environ["EP_XPRESS_SGLANG_SOURCE"])
        patch_file = ROOT / "adapters/sglang/sglang-v0.5.21.patch"
        files = [line[6:] for line in patch_file.read_text().splitlines() if line.startswith("+++ b/")]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for file in files:
                destination = temporary / file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / file, destination)
            subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=temporary, check=True, capture_output=True)
            subprocess.run(["git", "apply", str(patch_file)], cwd=temporary, check=True, capture_output=True)
            for file in files:
                if file.endswith(".py"):
                    ast.parse((temporary / file).read_text())


if __name__ == "__main__":
    unittest.main()
