# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Real two-block Wan integration; no checkpoint download or video generation."""
import json
import os
from pathlib import Path
import unittest
from types import SimpleNamespace

from PIL import Image

import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
from torch_npu.contrib import transfer_to_npu

from module.dit_cache.cache_method import NoCache, cache_manager
from wan.cache import first_block_forward
from wan.cache.taylorseer import WanTaylorSeer
from wan.modules.model import WanModel
from wan.image2video import WanI2V


class WanTaylorIntegrationTest(unittest.TestCase):
    def test_four_yaml_configs_use_wan_factory(self):
        previous = cache_manager.cache_method
        root = Path(__file__).resolve().parents[1]
        try:
            for name in ("nocache", "taylorseer", "taylorseer_offload", "taylorseer_prefetch"):
                cache_manager.from_config(str(root / "config" / f"14b_single_{name}.yaml"),
                                          cache_params={"num_steps": 40}, taylor_factory=WanTaylorSeer)
                if name == "nocache":
                    self.assertIsInstance(cache_manager.cache_method, NoCache)
                else:
                    self.assertIsInstance(cache_manager.cache_method, WanTaylorSeer)
                    self.assertEqual(cache_manager.cache_method.prefetch, name.endswith("prefetch"))
                    cache_manager.cache_method.cleanup()
        finally:
            cache_manager.cache_method = previous

    def test_real_generate_loop_uses_actual_device_and_resets(self):
        class TinyTextEncoder:
            model = torch.nn.Identity()

            def __call__(self, prompts, device):
                return [torch.zeros(4, 32, device=device, dtype=torch.bfloat16)]

        pipeline = WanI2V.__new__(WanI2V)
        pipeline.device = torch.device("cuda:0")  # same alias as the production constructor
        pipeline.init_on_cpu = True
        pipeline.t5_cpu = False
        pipeline.vae_stride = (4, 8, 8)
        pipeline.patch_size = (1, 2, 2)
        pipeline.sp_size = 1
        pipeline.rank = 0
        pipeline.use_sp = False
        pipeline.boundary = 0.9
        pipeline.num_train_timesteps = 1000
        pipeline.param_dtype = torch.bfloat16
        pipeline.sample_neg_prompt = "negative"
        pipeline.text_encoder = TinyTextEncoder()
        pipeline.vae = SimpleNamespace(
            encode=lambda _: [torch.zeros(16, 1, 4, 4, device="npu", dtype=torch.bfloat16)],
            decode=lambda x: x)
        for name in ("high_noise_model", "low_noise_model"):
            model = WanModel(model_type="i2v", dim=128, ffn_dim=256, freq_dim=32, text_dim=32,
                             text_len=4, in_dim=36, out_dim=16, num_heads=1, num_layers=2).to(torch.bfloat16)
            model.blocks[0].forward = first_block_forward.__get__(model.blocks[0], type(model.blocks[0]))
            setattr(pipeline, name, model.eval())
        cache = WanTaylorSeer({"enable_separate_cfg": True, "TaylorSeer": dict(
            cache_name="TaylorSeer", n_derivatives=1, warmup=1, skip_interval_steps=4,
            cutoff_steps=1, offload=True, prefetch=True, prefetch_distance=1)})
        previous = cache_manager.cache_method
        cache_manager.cache_method = cache
        try:
            image = Image.new("RGB", (32, 32))
            output = pipeline.generate("tiny", image, max_area=1024, frame_num=1, sampling_steps=8,
                                       guide_scale=3.5, seed=100)
            self.assertEqual(tuple(output.shape), (16, 1, 4, 4))
            audit = cache.last_statistics["audit"]
            self.assertEqual(len(audit), 16)
            self.assertEqual({(r["expert"], r["branch"]) for r in audit},
                             {("high", "cond"), ("high", "uncond"), ("low", "cond"), ("low", "uncond")})
            self.assertGreater(cache.last_statistics["predict_calls"], 0)
            self.assertFalse(cache.histories)
            pipeline.generate("tiny", image, max_area=1024, frame_num=1, sampling_steps=2,
                              guide_scale=3.5, seed=100)
            self.assertEqual(len(cache.last_statistics["audit"]), 4)
            self.assertFalse(cache.histories)
            self.assertFalse(cache.active)
        finally:
            cache.cleanup()
            cache_manager.cache_method = previous

    def test_real_wan_skips_remaining_block_and_preserves_modes(self):
        torch.manual_seed(123)
        model = WanModel(model_type="i2v", dim=128, ffn_dim=256, freq_dim=32, text_dim=32,
                         text_len=4, in_dim=36, out_dim=16, num_heads=1, num_layers=2)
        torch.nn.init.normal_(model.head.head.weight, std=0.02)
        model = model.to(device="npu:0", dtype=torch.bfloat16).eval()
        model.blocks[0].forward = first_block_forward.__get__(model.blocks[0], type(model.blocks[0]))
        inputs = dict(x=[torch.randn(16, 1, 4, 4, device="npu", dtype=torch.bfloat16)],
                      y=[torch.randn(20, 1, 4, 4, device="npu", dtype=torch.bfloat16)],
                      t=torch.tensor([900], device="npu"),
                      context=[torch.randn(4, 32, device="npu", dtype=torch.bfloat16)], seq_len=4)
        block_calls = []
        hook = model.blocks[1].register_forward_hook(lambda *_: block_calls.append(1))
        previous = cache_manager.cache_method
        results = []
        reference = None
        try:
            with torch.no_grad(), torch.amp.autocast("npu", dtype=torch.bfloat16):
                cache_manager.cache_method = NoCache()
                baseline = model(**inputs)[0].cpu()
                self.assertTrue(torch.isfinite(baseline).all())
                for offload, prefetch in ((False, False), (True, False), (True, True)):
                    config = {"enable_separate_cfg": True, "TaylorSeer": dict(
                        cache_name="TaylorSeer", n_derivatives=1, warmup=1, skip_interval_steps=3,
                        cutoff_steps=0, offload=offload, prefetch=prefetch, prefetch_distance=1)}
                    cache = WanTaylorSeer(config)
                    cache_manager.cache_method = cache
                    cache.begin_request(4)
                    block_calls.clear()
                    outputs = []
                    try:
                        for step in range(4):
                            for branch in ("cond", "uncond"):
                                cache.set_context(step, "high", branch, torch.device("npu:0"))
                                outputs.append(model(**inputs)[0].cpu())
                        self.assertEqual(len(block_calls), 4)
                        self.assertEqual(cache.compute_calls, 4)
                        self.assertEqual(cache.predict_calls, 4)
                        for output in outputs:
                            torch.testing.assert_close(output, baseline, rtol=1e-2, atol=1e-3)
                        if reference is not None:
                            for actual, expected in zip(outputs, reference):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        reference = outputs
                        results.append(dict(offload=offload, prefetch=prefetch, full_forwards=4,
                                            predicted_forwards=4, remaining_block_calls=len(block_calls),
                                            max_abs_vs_nocache=max((x - baseline).abs().max().item() for x in outputs)))
                    finally:
                        cache.cleanup()
                        self.assertFalse(cache.histories)
                        self.assertFalse(cache.staged)
        finally:
            hook.remove()
            cache_manager.cache_method = previous
        torch.npu.synchronize()
        output = os.environ.get("WAN_TEST_RESULT_JSON")
        if output:
            Path(output).write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
