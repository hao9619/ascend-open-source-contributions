# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Emit three API sessions; the separate checker verifies persisted filters and timestamp boundaries."""

import argparse
import json
from pathlib import Path
import time

import msmemscope
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def filtered_copy(x, output, block: tl.constexpr):
    index = tl.arange(0, block)
    tl.store(output + index, tl.load(x + index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "access", "both", "neither"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    x = torch.arange(256, device="npu", dtype=torch.float32)
    output = torch.zeros_like(x)
    filtered_copy[(1,)](x, output, 256)
    torch.npu.synchronize()
    events = {"launch": "launch", "access": "access", "both": "launch,access", "neither": "alloc,free"}
    msmemscope.config(events=events[args.mode], level="op", analysis="none", format="csv", output_path=args.output)
    windows = []
    for _ in range(3):
        msmemscope.start()
        before = time.time_ns()
        filtered_copy[(1,)](x, output, 256)
        after = time.time_ns()
        windows.append([before, after])
        torch.add(x, 1)
        torch.npu.synchronize()
        msmemscope.stop()
        filtered_copy[(1,)](x, output, 256)
        torch.npu.synchronize()
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    Path(args.output).mkdir(exist_ok=True, parents=True)
    expected = {"mode": args.mode, "windows": windows, "x": x.data_ptr(), "output": output.data_ptr(), "size": 1024}
    (Path(args.output) / "expected.json").write_text(json.dumps(expected))
    print(f"NATIVE FILTER APPLICATION PASS: {args.mode}", flush=True)


if __name__ == "__main__":
    main()
