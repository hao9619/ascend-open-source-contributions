# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Small tensor tests; run on CPU by default, or set WAN_TEST_DEVICE=npu:0."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location(
    "wan_taylorseer_unit", Path(__file__).resolve().parents[1] / "wan/cache/taylorseer.py")
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)
DEVICE = os.environ.get("WAN_TEST_DEVICE", "cpu")
if DEVICE.startswith("npu"):
    import torch_npu


def config(**overrides):
    params = dict(cache_name="TaylorSeer", n_derivatives=1, skip_interval_steps=4,
                  warmup=2, cutoff_steps=1, offload=False, prefetch=False)
    params.update(overrides)
    return {"TaylorSeer": params, "enable_separate_cfg": True}


class TaylorSeerTest(unittest.TestCase):
    def setUp(self):
        self.cache = ADAPTER.WanTaylorSeer(config())
        self.cache.begin_request(10)

    def tearDown(self):
        self.cache.cleanup()

    def forward(self, step, expert="high", branch="cond", offset=0, shape=(2, 3)):
        cache = self.cache
        cache.set_context(step, expert, branch, DEVICE)
        x = torch.full(shape, float(step), device=DEVICE)
        full, result = cache.pre_cache_process({"latent": x})
        if full:
            result = x + (2 * step + offset)
        cache.post_cache_update(result)
        return full, result.cpu(), x.cpu()

    def test_linear_residual_uses_real_steps_and_skips(self):
        for step in range(4):
            full, result, original = self.forward(step, offset=11)
            torch.testing.assert_close(result, original + 2 * step + 11)
        self.assertFalse(full)
        self.assertEqual(self.cache.compute_calls, 3)
        self.assertEqual(self.cache.predict_calls, 1)
        self.assertEqual(self.cache.histories[("high", "cond")].full_step, 2)

    def test_expert_and_cfg_histories_are_separate(self):
        for step in range(4):
            for branch, offset in (("cond", 10), ("uncond", 100)):
                _, result, original = self.forward(step, branch=branch, offset=offset)
                torch.testing.assert_close(result, original + 2 * step + offset)
        full, result, original = self.forward(4, expert="low", offset=1000)
        self.assertTrue(full)
        torch.testing.assert_close(result, original + 1008)
        self.assertEqual(len(self.cache.histories), 3)

    def test_interval_cutoff_and_per_expert_warmup(self):
        modes = [self.forward(step)[0] for step in range(10)]
        self.assertEqual(modes, [True, True, True, False, False, False, True, False, False, True])
        self.cache.cleanup()
        self.cache.begin_request(10)
        self.assertTrue(self.forward(7, expert="low")[0])
        self.assertTrue(self.forward(8, expert="low")[0])

    def test_shape_change_forces_new_history(self):
        for step in range(3):
            self.forward(step)
        self.assertTrue(self.forward(3, shape=(3, 3))[0])
        self.assertEqual(self.cache.audit[-1]["reason"], "shape_change")
        self.assertEqual(len(self.cache.histories[("high", "cond")].factors), 1)

    def test_full_stack_residual_not_first_block_residual(self):
        for step in range(4):
            self.cache.set_context(step, "high", "cond", DEVICE)
            x = torch.zeros((2, 3), device=DEVICE)
            full, result = self.cache.pre_cache_process({"latent": x})
            if full:
                result = result + 10  # first block
                result = result + 100 + 2 * step  # remaining blocks
            self.cache.post_cache_update(result)
            torch.testing.assert_close(result.cpu(), torch.full((2, 3), 110.0 + 2 * step))
        self.assertFalse(full)

    def test_request_reset_and_exception_cleanup(self):
        manager_module = SimpleNamespace(cache_manager=SimpleNamespace(cache_method=self.cache))
        self.cache.cleanup()

        @ADAPTER.taylor_cache_request
        def generate(sampling_steps=10, fail=False):
            self.forward(0)
            if fail:
                raise RuntimeError("injected failure")
            return self.cache.request_id

        with patch.dict(sys.modules, {"module.dit_cache.cache_method": manager_module}):
            first = generate(2)
            second = generate(10)
            self.assertGreater(second, first)
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                generate(fail=True)
        self.assertFalse(self.cache.active)
        self.assertFalse(self.cache.histories)
        self.assertFalse(self.cache.staged)
        self.assertIsNone(self.cache.original)
        self.assertIsNone(self.cache.context)

    def test_repeated_step_and_missing_context_rejected(self):
        self.forward(0)
        with self.assertRaises(ValueError):
            self.forward(0)
        with self.assertRaises(RuntimeError):
            self.cache.pre_cache_process({"latent": torch.zeros(1)})

    def test_orders_zero_through_three_use_finite_differences(self):
        for order, expected_residual in ((0, 27), (1, 46), (2, 52), (3, 53)):
            self.cache.cleanup()
            self.cache = ADAPTER.WanTaylorSeer(config(n_derivatives=order, warmup=3))
            self.cache.begin_request(10)
            for step in range(5):
                self.cache.set_context(step, "high", "cond", DEVICE)
                x = torch.zeros((2, 3), device=DEVICE)
                full, result = self.cache.pre_cache_process({"latent": x})
                if full:
                    result = x + step ** 3
                self.cache.post_cache_update(result)
            self.assertFalse(full)
            self.assertEqual(len(self.cache.histories[("high", "cond")].factors), order + 1)
            torch.testing.assert_close(result.cpu(), torch.full((2, 3), float(expected_residual)))

    def test_dtype_change_forces_full_and_output_dtype_is_preserved(self):
        for step in range(3):
            self.forward(step)
        self.cache.set_context(3, "high", "cond", DEVICE)
        x = torch.zeros((2, 3), device=DEVICE, dtype=torch.bfloat16)
        full, _ = self.cache.pre_cache_process({"latent": x})
        self.assertTrue(full)
        self.cache.post_cache_update(x.float() + 3.125)
        self.cache.set_context(4, "high", "cond", DEVICE)
        full, _ = self.cache.pre_cache_process({"latent": x})
        self.assertTrue(full)  # per-history warmup restarts after signature change
        self.cache.post_cache_update(x.float() + 3.125)
        self.cache.set_context(5, "high", "cond", DEVICE)
        full, result = self.cache.pre_cache_process({"latent": x})
        self.assertFalse(full)
        self.assertEqual(result.dtype, torch.float32)
        self.cache.post_cache_update(result)

    def test_prefetch_cleanup_with_pending_copy(self):
        self.cache.cleanup()
        self.cache = ADAPTER.WanTaylorSeer(config(offload=True, prefetch=True))
        self.cache.begin_request(10)
        for step in range(2):
            for branch in ("cond", "uncond"):
                self.forward(step, branch=branch)
        self.cache.set_context(2, "high", "cond", DEVICE)
        if DEVICE.startswith("npu"):
            self.assertTrue(self.cache.staged)
            self.assertIsNotNone(self.cache.copy_stream)
        self.cache.cleanup()
        self.assertFalse(self.cache.staged)
        self.assertFalse(self.cache.histories)
        self.assertIsNone(self.cache.copy_stream)

    def test_invalid_configuration(self):
        invalid = [dict(prefetch=True), dict(offload="false"), dict(prefetch=1), dict(warmup=0),
                   dict(skip_interval_steps=0), dict(skip_interval_steps=True), dict(cutoff_steps=-1),
                   dict(n_derivatives=4), dict(typo=True), dict(prefetch_distance=0),
                   dict(prefetch_distance=2), dict(prefetch_distance=True)]
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValueError):
                ADAPTER.WanTaylorSeer(config(**params))
        cfg = config()
        cfg["enable_separate_cfg"] = False
        with self.assertRaises(ValueError):
            ADAPTER.WanTaylorSeer(cfg)

    def test_offload_and_prefetch_match_resident_results(self):
        for prefetch in (False, True):
            self.cache.cleanup()
            self.cache = ADAPTER.WanTaylorSeer(config(offload=True, prefetch=prefetch))
            self.cache.begin_request(10)
            for step in range(7):
                for branch, offset in (("cond", 10), ("uncond", 100)):
                    _, result, original = self.forward(step, branch=branch, offset=offset)
                    torch.testing.assert_close(result, original + 2 * step + offset)
            self.assertTrue(all(t.device.type == "cpu" for h in self.cache.histories.values() for t in h.factors))
            self.cache.cleanup()
            self.assertFalse(self.cache.staged)
            self.assertIsNone(self.cache.copy_stream)


if __name__ == "__main__":
    unittest.main(verbosity=2)
