# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Collect a Triton copy and an in-place update using the standard MemScope API."""

import argparse

import msmemscope
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def copy_and_increment(x, output, size: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.load(x + index, index < size, 0)
    tl.store(output + index, value + 1, index < size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="./triton_trace")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    msmemscope.config(
        events="alloc,free,launch,access", level="op", analysis="none", format="csv", output_path=args.output
    )
    torch.npu.set_device(args.device)
    x = torch.arange(1025, device="npu", dtype=torch.float32)
    output = torch.empty_like(x)
    msmemscope.start()
    try:
        copy_and_increment[(triton.cdiv(x.numel(), 256),)](x, output, x.numel(), 256)
        copy_and_increment[(triton.cdiv(x.numel(), 256),)](output, output, output.numel(), 256)
        torch.npu.synchronize()
    finally:
        msmemscope.stop()
    torch.testing.assert_close(output, x + 2)
    print("Triton output verified; collection written to", args.output)


if __name__ == "__main__":
    main()
