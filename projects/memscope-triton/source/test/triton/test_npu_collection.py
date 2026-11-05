# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Real NPU functional/IR tests. Run with the built msmemscope package on PYTHONPATH."""

import argparse

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from msmemscope.triton_access import analyze_kernel
from msmemscope.triton_collection import TritonCollector


@triton.jit
def copy_kernel(x, output, n: tl.constexpr, stride: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    tl.store(output + i, tl.load(x + i * stride, i < n, 0), i < n)


@triton.jit
def inplace_kernel(x, n: tl.constexpr, block: tl.constexpr):
    i = tl.arange(0, block)
    tl.store(x + i, tl.load(x + i, i < n, 0) + 1, i < n)


@triton.jit
def atomic_kernel(x, output, n: tl.constexpr, block: tl.constexpr):
    i = tl.arange(0, block)
    tl.atomic_add(output + i, tl.load(x + i, i < n, 0), i < n)


@triton.jit
def loop_kernel(x, output, steps, block: tl.constexpr):
    i = tl.arange(0, block)
    pointer = x + i
    total = tl.full((block,), 0, tl.float32)
    for _ in range(steps):
        total += tl.load(pointer)
        pointer += block
    tl.store(output + i, total)


def check_effects(kernel, expected):
    effects = analyze_kernel(kernel)
    assert [(effect.read, effect.write, effect.unknown) for effect in effects] == expected


def check_copy_cases(collector, events):
    count = 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32):
        for size in (0, 1, 17, 257, 1025):
            for stride in (1, 2):
                x = torch.arange(size * stride, dtype=torch.int32, device="npu").to(dtype)[::stride]
                output = torch.empty(size, dtype=dtype, device="npu")
                before = len(events)
                kernel = copy_kernel[(max(1, triton.cdiv(size, 256)),)](x, output, size, stride, 256)
                torch.npu.synchronize()
                torch.testing.assert_close(output, x, rtol=0, atol=0)
                check_effects(kernel, [(True, False, False), (False, True, False)])
                begin, end = events[before:]
                assert begin["call_id"] == end["call_id"]
                if size == 0:
                    assert begin["accesses"] == []
                else:
                    read, write = begin["accesses"]
                    assert read["ptr"] == x.data_ptr()
                    assert read["size"] == ((size - 1) * stride + 1) * x.element_size()
                    assert write["ptr"] == output.data_ptr()
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    collector = TritonCollector()
    events = []
    collector._emit = events.append
    collector.enable()
    try:
        count = check_copy_cases(collector, events)
        x = torch.ones(768, device="npu")
        output = torch.zeros(256, device="npu")
        kernel = loop_kernel[(1,)](x, output, 3, 256)
        check_effects(kernel, [(True, False, False), (False, True, False), (False, False, False)])
        torch.testing.assert_close(output, torch.full_like(output, 3))
        kernel = atomic_kernel[(1,)](x, output, 256, 256)
        check_effects(kernel, [(True, False, False), (True, True, False)])
        torch.testing.assert_close(output, torch.full_like(output, 4))
        kernel = inplace_kernel[(1,)](output, 256, 256)
        check_effects(kernel, [(True, True, False)])
        torch.testing.assert_close(output, torch.full_like(output, 5))
        before = len(events)
        copy_kernel[(1,)](output, output, 256, 1, 256)
        read, write = events[before]["accesses"]
        assert read["ptr"] == write["ptr"] == output.data_ptr()
        assert read["read"] and write["write"]
        torch.npu.synchronize()
        print(f"NPU PASS: {count} copy parameter combinations and 4 loop/atomic/inplace/alias cases", flush=True)
    finally:
        collector.disable()


if __name__ == "__main__":
    main()
