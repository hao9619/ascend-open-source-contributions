# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Focused NPU checks for conditional effects, indexed pointers, views, and failure isolation."""

import argparse
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from msmemscope.triton_access import analyze_kernel
from msmemscope.triton_collection import TritonCollector


@triton.jit
def branch_access(x, y, output, choose, block: tl.constexpr):
    i = tl.arange(0, block)
    if choose > 0:
        tl.store(output + i, tl.load(x + i))
    else:
        tl.store(output + i, tl.load(y + i))


@triton.jit
def indexed_access(x, indices, output, block: tl.constexpr):
    i = tl.arange(0, block)
    index = tl.load(indices + i)
    tl.store(output + i, tl.load(x + index))


@triton.jit
def view_access(x, output, stride0: tl.constexpr, stride1: tl.constexpr):
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, 16)[None, :]
    tl.store(output + row * 16 + col, tl.load(x + row * stride0 + col * stride1))


@triton.jit
def optional_write(x, output, write: tl.constexpr, block: tl.constexpr):
    i = tl.arange(0, block)
    if write:
        tl.store(output + i, tl.load(x + i))


def effects(kernel):
    result = [(item.read, item.write, item.unknown) for item in analyze_kernel(kernel)]
    print("COMPILED EFFECTS", kernel.name, kernel.src.signature, kernel.src.constants, result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("branch", "indexed", "view", "specialization", "failure"))
    args = parser.parse_args()
    torch.npu.set_device(0)
    x = torch.arange(256, device="npu", dtype=torch.float32)
    y = x + 7
    output = torch.zeros_like(x)
    collector = TritonCollector()
    events = []
    collector._emit = events.append
    collector.enable()
    read, write, none = (True, False, False), (False, True, False), (False, False, False)
    try:
        if args.case == "branch":
            for choose in (1, -1, 2, -2):
                kernel = branch_access[(1,)](x, y, output, choose, 256)
                # Triton specializes the integer value 1 and removes its runtime IR argument.
                expected = [read, none, write] if choose == 1 else [read, read, write, none]
                assert effects(kernel) == expected
                torch.testing.assert_close(output, x if choose > 0 else y, rtol=0, atol=0)
        elif args.case == "indexed":
            for index in (torch.arange(255, -1, -1, device="npu"), torch.zeros(256, device="npu", dtype=torch.int64)):
                kernel = indexed_access[(1,)](x, index, output, 256)
                assert effects(kernel) == [read, read, write]
                torch.testing.assert_close(output, x[index], rtol=0, atol=0)
        elif args.case == "view":
            storage = torch.arange(1024, device="npu", dtype=torch.float32)
            for view in (storage[:256].reshape(16, 16).t(), storage[1:513:2].reshape(16, 16)):
                kernel = view_access[(1,)](view, output, *view.stride())
                assert effects(kernel) == [read, write]
                torch.testing.assert_close(output.reshape(16, 16), view, rtol=0, atol=0)
                access = events[-2]["accesses"][0]
                expected = (1 + sum((n - 1) * s for n, s in zip(view.shape, view.stride()))) * 4
                assert (access["ptr"], access["size"]) == (view.data_ptr(), expected)
        elif args.case == "specialization":
            for enabled in (True, False, True):
                kernel = optional_write[(1,)](x, output, enabled, 256)
                assert effects(kernel) == ([read, write] if enabled else [none, none])
                assert len(events[-2]["accesses"]) == (2 if enabled else 0)
        else:
            with patch(
                "msmemscope.triton_collection.analyze_kernel", side_effect=ValueError("injected analysis error"),
            ):
                optional_write[(1,)](x, output, True, 256)
            torch.testing.assert_close(output, x, rtol=0, atol=0)
            assert all(a["unknown"] and not a["read"] and not a["write"] for a in events[-2]["accesses"])
            before = len(events)
            try:
                optional_write[(1,)](x)
            except TypeError:
                pass
            else:
                raise AssertionError("Missing arguments did not raise")
            assert len(events) == before and not collector._local.stack
            with patch.object(collector, "_emit", side_effect=RuntimeError("injected transport error")):
                optional_write[(1,)](x, output, True, 256)
            assert not collector._local.stack
            optional_write[(1,)](x, output, True, 256)
            torch.testing.assert_close(output, x, rtol=0, atol=0)
        torch.npu.synchronize()
        assert len(events) % 2 == 0
        for begin, end in zip(events[::2], events[1::2]):
            assert begin["phase"] == "begin" and end["phase"] == "end"
            assert begin["call_id"] == end["call_id"]
        print(f"ACCEPTANCE CASE PASS: {args.case}; captured launches={len(events) // 2}", flush=True)
    finally:
        collector.disable()


if __name__ == "__main__":
    main()
