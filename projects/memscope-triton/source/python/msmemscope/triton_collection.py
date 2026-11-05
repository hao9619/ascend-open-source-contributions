# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Optional Triton-Ascend 3.2 launch collection using compiled parameter memory effects."""

import functools
import itertools
import json
import logging
import threading
import weakref

from .triton_access import ParameterAccess, analyze_kernel

_LOG = logging.getLogger(__name__)
_KEY = "_memscope_triton"
_UINT64_MAX = (1 << 64) - 1


def tensor_extent(tensor):
    """Return the parameter's bounding storage span, including holes in a strided view."""
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    if any(dimension == 0 for dimension in shape):
        return None
    if any(stride < 0 for stride in strides):
        raise ValueError("Negative strides are unsupported")
    address = int(tensor.data_ptr())
    size = (1 + sum((dimension - 1) * stride for dimension, stride in zip(shape, strides))) * tensor.element_size()
    if address <= 0 or size <= 0 or size > (1 << 63) - 1 or address > _UINT64_MAX - size:
        raise ValueError("Invalid or overflowing parameter extent")
    storage = tensor.untyped_storage()
    storage_start = storage.data_ptr()
    if address < storage_start or address + size > storage_start + storage.nbytes():
        raise ValueError("Parameter span exceeds storage bounds")
    return address, size


class TritonCollector:
    """Install reversible launch hooks; retain compiled analysis but never Tensor arguments."""

    def __init__(self):
        self.enabled = False
        self._lock = threading.RLock()
        self._cache = weakref.WeakKeyDictionary()
        self._patches = []
        self._warned = set()
        self._calls = itertools.count()
        self._local = threading.local()

    def _warn(self, message):
        if message not in self._warned:
            self._warned.add(message)
            _LOG.warning("Triton collection: %s", message)

    def _payload(self, kernel, stream, args):
        import torch

        function = getattr(kernel.src, "fn", None)
        if function is None:
            raise ValueError("Only JIT ASTSource kernels are supported")
        names = [name for index, name in enumerate(function.arg_names) if index not in function.constexprs]
        if len(names) != len(args):
            raise ValueError("Runtime parameter binding does not match the compiled signature")
        with self._lock:
            effects = self._cache.get(kernel)
            if effects is None:
                ir_names = [name for name in kernel.src.signature if name not in kernel.src.constants]
                try:
                    inferred = analyze_kernel(kernel)
                except Exception as error:
                    self._warn(f"IR analysis unavailable: {error}")
                    inferred = [ParameterAccess(unknown=True) for _ in ir_names]
                if len(ir_names) != len(inferred):
                    raise ValueError("Compiled IR parameter mapping is unsupported")
                effects = dict(zip(ir_names, inferred))
                self._cache[kernel] = effects
        accesses = []
        devices = set()
        for name, value in zip(names, args):
            if not isinstance(value, torch.Tensor) or value.device.type != "npu":
                if str(kernel.src.signature.get(name, "")).startswith("*") and value is not None:
                    self._warn(f"Parameter {name} has no supported NPU Tensor storage extent")
                continue
            effect = effects.get(name, ParameterAccess(unknown=True))
            if not (effect.read or effect.write or effect.unknown):
                continue
            extent = tensor_extent(value)
            if extent is None:
                continue
            devices.add(value.device.index)
            address, size = extent
            accesses.append(
                {
                    "parameter": name,
                    "ptr": address,
                    "size": size,
                    "read": effect.read,
                    "write": effect.write,
                    "unknown": effect.unknown,
                }
            )
        if len(devices) > 1:
            raise ValueError("A single launch refers to multiple NPU devices")
        return {
            "name": kernel.name,
            "device": next(iter(devices)) if devices else torch.npu.current_device(),
            "stream_handle": int(stream),
            "call_id": next(self._calls),
            "accesses": accesses,
            "extent_kind": "parameter_storage_span",
            "effect_kind": "static_possible",
        }

    def _emit(self, payload):
        import mstx

        mstx.mark("memscope-triton:" + json.dumps(payload, separators=(",", ":")), None)

    def _enter(self, metadata):
        stack = getattr(self._local, "stack", None)
        if stack is None:
            self._local.stack = stack = []
        stack.append((metadata, None))
        try:
            if not self.enabled or metadata is None:
                return
            payload = metadata.get().get(_KEY)
            if payload is None:
                return
            self._emit(dict(payload, phase="begin"))
            stack[-1] = (metadata, {key: value for key, value in payload.items() if key != "accesses"})
        except Exception as error:
            self._warn(str(error))

    def _exit(self, metadata=None):
        stack = getattr(self._local, "stack", [])
        if not stack:
            return
        if metadata is not None and stack[-1][0] is not metadata:
            return
        _, payload = stack.pop()
        if payload is None:
            return
        try:
            self._emit(dict(payload, phase="end"))
        except Exception as error:
            self._warn(str(error))

    def _install(self, owner, name, wrapper):
        original = getattr(owner, name)
        replacement = wrapper(original)
        self._patches.append((owner, name, original, replacement))
        setattr(owner, name, replacement)

    def enable(self):
        with self._lock:
            if self.enabled:
                return
            try:
                from triton.compiler.compiler import CompiledKernel
                from triton.runtime.jit import JITFunction
            except ImportError:
                return
            if not hasattr(CompiledKernel, "launch_enter_hook"):
                self._warn("This Triton version has no supported CompiledKernel launch hooks")
                return

            def metadata_wrapper(original):
                @functools.wraps(original)
                def wrapped(kernel, grid, stream, *args):
                    metadata = original(kernel, grid, stream, *args)
                    if self.enabled and metadata is not None:

                        def extra():
                            try:
                                return {_KEY: self._payload(kernel, stream, args)}
                            except Exception as error:
                                self._warn(str(error))
                                return {}

                        metadata.add(extra, ())
                    return metadata

                return wrapped

            def enter_wrapper(original):
                def wrapped(metadata):
                    if original is not None:
                        original(metadata)
                    self._enter(metadata)

                return wrapped

            def exit_wrapper(original):
                def wrapped(metadata):
                    try:
                        self._exit(metadata)
                    finally:
                        if original is not None:
                            original(metadata)

                return wrapped

            def invocation_wrapper(original):
                @functools.wraps(original)
                def wrapped(*args, **kwargs):
                    depth = len(getattr(self._local, "stack", []))
                    try:
                        return original(*args, **kwargs)
                    finally:
                        while len(getattr(self._local, "stack", [])) > depth:
                            self._exit()

                return wrapped

            def runner_wrapper(original):
                @functools.wraps(original)
                def wrapped(kernel, grid):
                    return invocation_wrapper(original(kernel, grid))

                return wrapped

            self._install(CompiledKernel, "launch_metadata", metadata_wrapper)
            self._install(CompiledKernel, "launch_enter_hook", enter_wrapper)
            self._install(CompiledKernel, "launch_exit_hook", exit_wrapper)
            self._install(JITFunction, "run", invocation_wrapper)
            self._install(CompiledKernel, "__getitem__", runner_wrapper)
            self.enabled = True

    def disable(self):
        with self._lock:
            self.enabled = False
            for owner, name, original, replacement in reversed(self._patches):
                if getattr(owner, name) is replacement:
                    setattr(owner, name, original)
            self._patches.clear()
            self._cache.clear()


triton_collector = TritonCollector()


def enable_triton_collector():
    triton_collector.enable()


def disable_triton_collector():
    triton_collector.disable()
