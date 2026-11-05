# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Boundary checks for the medium large-batch vector triangular-inverse path."""

import argparse
import ctypes as ct
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tests_root))
    from validate import Solver
    from validate_batch import Reference, run_case

    args.output.mkdir(parents=True, exist_ok=False)
    solver = Solver(args.library, 0)
    reference = Reference(args.reference_bridge)
    rows = []

    def record(row):
        rows.append(row)
        print(json.dumps(row), flush=True)
        solver.release()
        if not row["passed"]:
            raise RuntimeError("Medium boundary failure")

    try:
        index = 6000
        for n in (33, 57, 60, 64, 65, 66, 96, 127, 128):
            for pad in (0, 8, 32):
                record(
                    run_case(
                        solver,
                        reference,
                        "sgetribatched",
                        dict(n=n, batch=4096, lda=n + pad, ldc=n + pad),
                        index,
                        args.output,
                        0,
                        2,
                    )
                )
                index += 1
        for n, pad, pivot in ((57, 8, True), (66, 32, False)):
            batch, ld = 4096, n + pad
            # Strided inner pointers ensure the generic Device pointer path is exercised.
            source = np.full((2 * batch, n, ld), -777, np.float32)
            source[::2, :, :n] = np.eye(n, dtype=np.float32) * 2
            source[10, n - 1, n - 1] = 0
            result = np.full_like(source, -999)
            expected = result.copy()
            pp = np.tile(np.arange(1, n + 1, dtype=np.int32), (batch, 1))
            if pivot:
                pp[6, -1] = 0
            original = source.copy()
            a, c = solver.upload(source), solver.upload(result)
            offsets = np.arange(batch, dtype=np.uint64) * 2 * n * ld * 4
            apointers = np.asarray(a.value + offsets, np.uint64)
            cpointers = np.asarray(c.value + offsets, np.uint64)
            apointers[2] = 0
            cpointers[3] = 0
            cpointers[4] = apointers[4]
            ap, cp = solver.upload(apointers), solver.upload(cpointers)
            p = solver.upload(pp) if pivot else ct.c_void_p()
            info = solver.allocate(batch * 4)
            expected_info = np.zeros(batch, np.int32)
            expected_info[2:5] = [-2, -5, -5]
            expected_info[5] = n
            if pivot:
                expected_info[6] = -4
            for i in np.flatnonzero(expected_info == 0):
                expected[i * 2, :, :n] = np.eye(n, dtype=np.float32) * 0.5
            first = None
            deterministic = True
            for _ in range(2):
                solver.call(
                    "aclsolverSgetriBatched",
                    solver.handle,
                    ct.c_int(n),
                    ap,
                    ct.c_int(ld),
                    p,
                    cp,
                    ct.c_int(ld),
                    info,
                    ct.c_int(batch),
                )
                solver.sync()
                solver.download(c, result)
                if first is None:
                    first = result.copy()
                else:
                    deterministic = np.array_equal(first, result)
            observed = solver.download(info, np.empty(batch, np.int32))
            received = solver.download(a, np.empty_like(source))
            pivots_ok = not pivot or np.array_equal(
                solver.download(p, np.empty_like(pp)), pp
            )
            passed = bool(
                np.array_equal(result, expected)
                and np.array_equal(observed, expected_info)
                and np.array_equal(received, original)
                and deterministic
                and pivots_ok
            )
            record(
                dict(
                    test="strided_info_no_pivot" if not pivot else "strided_info_pivot",
                    n=n,
                    batch=batch,
                    lda=ld,
                    pivot=pivot,
                    deterministic=deterministic,
                    passed=passed,
                    info_mismatches=int(np.count_nonzero(observed != expected_info)),
                    output_mismatches=int(np.count_nonzero(result != expected)),
                )
            )
    finally:
        solver.close()
        (args.output / "summary.json").write_text(
            json.dumps(
                dict(
                    tests=len(rows),
                    passed=sum(r["passed"] for r in rows),
                    results=rows,
                    library_sha256=hashlib.sha256(
                        args.library.read_bytes()
                    ).hexdigest(),
                    reference_sha256=hashlib.sha256(
                        reference.path.read_bytes()
                    ).hexdigest(),
                    bridge_sha256=hashlib.sha256(
                        args.reference_bridge.read_bytes()
                    ).hexdigest(),
                    script_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                ),
                indent=2,
            )
        )
    return int(len(rows) != 29 or not all(row["passed"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
