# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Request-local Taylor prediction of the complete Wan block-stack residual.

Differences use actual denoising step distances, following the finite-difference
recurrence in module/dit_cache/cache_method.py, not a CFG-forward call counter.
"""
import inspect
import logging
import math
from dataclasses import dataclass, field
from functools import wraps

import torch


@dataclass
class _History:
    factors: list = field(default_factory=list)
    full_step: int = -1
    last_step: int = -1
    visits: int = 0
    input_dtype: object = None
    output_dtype: object = None


class WanTaylorSeer:
    """Wan-only adapter; histories never cross expert, branch, or request."""

    cache_name = "TaylorSeer"

    def __init__(self, config, cache_params=None):
        params = dict(config["TaylorSeer"])
        allowed = {"cache_name", "n_derivatives", "skip_interval_steps", "warmup", "cutoff_steps",
                   "offload", "prefetch", "prefetch_distance"}
        if params.get("cache_name") != "TaylorSeer":
            raise ValueError("Wan TaylorSeer cache_name must be TaylorSeer")
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"Unknown Wan TaylorSeer parameters: {sorted(unknown)}")
        if config.get("enable_separate_cfg") is not True:
            raise ValueError("Wan TaylorSeer requires enable_separate_cfg=true")
        for name, minimum, maximum in (("n_derivatives", 0, 3), ("skip_interval_steps", 1, None),
                                       ("warmup", 1, None), ("cutoff_steps", 0, None)):
            value = params[name]
            if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f"Invalid Wan TaylorSeer {name}: {value!r}")
            setattr(self, name, value)
        for name in ("offload", "prefetch"):
            value = params.get(name, False)
            if type(value) is not bool:
                raise ValueError(f"Wan TaylorSeer {name} must be a boolean")
            setattr(self, name, value)
        self.prefetch_distance = params.get("prefetch_distance", 1)
        if type(self.prefetch_distance) is not int or self.prefetch_distance != 1:
            raise ValueError("Wan TaylorSeer supports only prefetch_distance=1")
        if self.prefetch and not self.offload:
            raise ValueError("Wan TaylorSeer prefetch requires offload=true")
        self.should_skip = False
        self.active = False
        self.histories = {}
        self.staged = {}
        self.copy_stream = None
        self.context = None
        self.original = None
        self.audit = []
        self.last_statistics = {}
        self.request_id = 0
        self.total_steps = 0
        self.compute_calls = 0
        self.predict_calls = 0

    def begin_request(self, total_steps):
        if self.active:
            raise RuntimeError("Concurrent or nested Wan TaylorSeer requests are unsupported")
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("total_steps must be a positive integer")
        self.cleanup()
        self.request_id += 1
        self.total_steps = total_steps
        self.compute_calls = self.predict_calls = 0
        self.audit = []
        self.active = True

    def set_context(self, step, expert, branch, device):
        if not self.active:
            raise RuntimeError("Wan TaylorSeer requires an active request")
        if type(step) is not int or not 0 <= step < self.total_steps:
            raise ValueError("Invalid denoising step")
        if expert not in ("high", "low") or branch not in ("cond", "uncond"):
            raise ValueError("Invalid expert or CFG branch")
        if self.context is not None:
            raise RuntimeError("Previous Wan TaylorSeer forward was not completed")
        key = (expert, branch)
        history = self.histories.setdefault(key, _History())
        if step <= history.last_step:
            raise ValueError("Denoising steps must increase within each expert/CFG history")
        self.context = (key, step, torch.device(device))
        self.should_skip = False
        if self.prefetch:
            self._prefetch(key, torch.device(device))
            # The unconditional cache is transferred while conditional embeddings/blocks run.
            if branch == "cond":
                self._prefetch((expert, "uncond"), torch.device(device))

    def _prefetch(self, key, device):
        history = self.histories.get(key)
        if not history or not history.factors or key in self.staged or device.type == "cpu":
            return
        backend = getattr(torch, device.type)
        if self.copy_stream is None:
            self.copy_stream = backend.Stream(device=device)
        with backend.stream(self.copy_stream):
            tensors = [factor.to(device=device, non_blocking=True) for factor in history.factors]
            ready = backend.Event()
            ready.record(self.copy_stream)
        # Keep pinned sources alive until the H2D event has actually completed.
        self.staged[key] = (tensors, ready, history.factors)

    def _load_factors(self, key, device):
        if key not in self.staged:
            return [factor.to(device=device) for factor in self.histories[key].factors]
        tensors, ready, _sources = self.staged[key]
        current = getattr(torch, device.type).current_stream(device)
        current.wait_event(ready)
        for tensor in tensors:
            tensor.record_stream(current)
        return tensors

    def _release_stage(self, key):
        staged = self.staged.pop(key, None)
        if staged is not None:
            staged[1].synchronize()

    def _reason_for_full(self, history, step, latent):
        if not history.factors:
            return "empty_history"
        factor = history.factors[0]
        if factor.shape != latent.shape or history.input_dtype != latent.dtype:
            return "shape_change"
        if history.visits < self.warmup:
            return "expert_branch_warmup"
        if step < self.warmup:
            return "request_warmup"
        if self.total_steps - step <= self.cutoff_steps:
            return "cutoff"
        if (step - self.warmup) % self.skip_interval_steps == 0:
            return "interval"
        return None

    def pre_cache_process(self, args):
        if self.context is None:
            raise RuntimeError("Missing explicit Wan TaylorSeer step/expert/CFG context")
        latent = args["latent"]
        key, step, device = self.context
        history = self.histories[key]
        reason = self._reason_for_full(history, step, latent)
        if reason == "shape_change":
            self._release_stage(key)
            history.factors = []
            history.full_step = -1
            history.visits = 0
        self.audit.append({"request": self.request_id, "step": step, "expert": key[0], "branch": key[1],
                           "mode": "full" if reason else "predict", "reason": reason or "scheduled_prediction"})
        history.last_step = step
        history.visits += 1
        if reason:
            self.original = latent.detach().clone()
            self.compute_calls += 1
            return True, latent
        factors = self._load_factors(key, device)
        distance = step - history.full_step
        residual = torch.zeros_like(factors[0])
        for order, factor in enumerate(factors):
            residual = residual + factor * (distance ** order / math.factorial(order))
        self.predict_calls += 1
        self.should_skip = True
        return False, (latent.float() + residual).to(dtype=history.output_dtype)

    def _store_factors(self, factors, device):
        if not self.offload:
            return [factor.detach() for factor in factors]
        stored = []
        for factor in factors:
            # Blocking D2H is deliberate: pinned history is immutable before any prefetch reads it.
            host = factor.detach().to(device="cpu")
            if self.prefetch and device.type != "cpu":
                host = host.pin_memory(device=device.type)
            stored.append(host)
        return stored

    def post_cache_update(self, latent):
        if self.context is None:
            raise RuntimeError("Missing Wan TaylorSeer forward context at update")
        key, step, device = self.context
        history = self.histories[key]
        if not self.should_skip:
            if self.original is None:
                raise RuntimeError("Full Wan TaylorSeer forward did not capture its input")
            factors = [(latent.float() - self.original.float()).detach()]
            previous = self._load_factors(key, device)
            if previous:
                distance = step - history.full_step
                if distance <= 0:
                    raise RuntimeError("Non-positive Taylor finite-difference distance")
                for order in range(min(self.n_derivatives, len(previous))):
                    factors.append((factors[order] - previous[order]) / distance)
            self._release_stage(key)
            history.factors = self._store_factors(factors, device)
            history.full_step = step
            history.input_dtype = self.original.dtype
            history.output_dtype = latent.dtype
        else:
            self._release_stage(key)
        self.original = None
        self.context = None
        self.should_skip = False

    def cleanup(self):
        try:
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
        finally:
            self.last_statistics = {"request": self.request_id, "compute_calls": self.compute_calls,
                                    "predict_calls": self.predict_calls, "audit": list(self.audit)}
            self.histories.clear()
            self.staged.clear()
            self.context = self.original = self.copy_stream = None
            self.should_skip = False
            self.active = False

    def print_statistics(self):
        stats = self.last_statistics
        logging.info("Wan TaylorSeer request=%s full_forwards=%s predicted_forwards=%s",
                     stats.get("request"), stats.get("compute_calls"), stats.get("predict_calls"))


def taylor_cache_request(function):
    """Reset history for every generate call, including warmup, and clean up on errors."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        from module.dit_cache.cache_method import cache_manager
        cache = cache_manager.cache_method
        if not isinstance(cache, WanTaylorSeer):
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        cache.begin_request(bound.arguments["sampling_steps"])
        try:
            return function(*args, **kwargs)
        finally:
            cache.cleanup()
    return wrapped
