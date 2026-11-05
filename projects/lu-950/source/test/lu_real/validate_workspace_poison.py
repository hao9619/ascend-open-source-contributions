# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check that caller-workspace status bytes never affect a valid inversion."""

import argparse
import ctypes as ct
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import Solver  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--n", type=int, default=150)
    args = parser.parse_args()
    solver = Solver(args.library, 0)
    rows = []
    try:
        n = args.n
        ld = (n + 7) // 8 * 8
        initial = np.eye(n, dtype=np.float32, order="F") * 2
        matrix = solver.upload(initial)
        info = solver.allocate(4)
        count = ct.c_int()
        solver.call(
            "aclsolverSgetri_bufferSize",
            solver.handle,
            ct.c_int(n),
            matrix,
            ct.c_int(n),
            ct.byref(count),
        )
        workspace = solver.allocate(count.value * 4)
        status_offset = (4 * n * ld * 4 + 511) // 512 * 512
        status_pointer = ct.c_void_p(workspace.value + status_offset)
        for repeat in range(12):
            solver.write(matrix, initial)
            solver.call(
                "aclrtMemset",
                status_pointer,
                ct.c_size_t(512),
                ct.c_int(255),
                ct.c_size_t(512),
                runtime=True,
            )
            solver.call(
                "aclsolverSgetri",
                solver.handle,
                ct.c_int(n),
                matrix,
                ct.c_int(n),
                ct.c_void_p(),
                workspace,
                count,
                info,
            )
            solver.sync()
            output = solver.download(matrix, np.empty_like(initial))
            observed_info = int(solver.download(info, np.empty(1, np.int32))[0])
            statuses = solver.download(status_pointer, np.empty(128, np.int32))
            entries = statuses[: 5 * args.stride : args.stride].tolist()
            passed = bool(
                observed_info == 0
                and np.array_equal(output, np.eye(n, dtype=np.float32) / 2)
            )
            rows.append(
                dict(
                    repeat=repeat,
                    info=observed_info,
                    statuses=entries,
                    mismatch_count=int(
                        np.count_nonzero(output != np.eye(n, dtype=np.float32) / 2)
                    ),
                    passed=passed,
                )
            )
    finally:
        solver.close()
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                repeats=len(rows),
                passed=sum(row["passed"] for row in rows),
                results=rows,
            )
        ),
        flush=True,
    )
    return int(not all(row["passed"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
