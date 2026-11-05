# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Representative Device memory coverage and resident-HBM observations; never a performance timer."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tests_root))
    from validate import Solver

    solver = Solver(args.library, 0)
    rows = []

    def free_hbm():
        free, total = ct.c_size_t(), ct.c_size_t()
        # ACL_HBM_MEM=1 verified in the installed CANN9.1.0 acl_rt.h.
        solver.call(
            "aclrtGetMemInfo",
            ct.c_int(1),
            ct.byref(free),
            ct.byref(total),
            runtime=True,
        )
        return free.value, total.value

    def record(row, baseline):
        free, total = free_hbm()
        row.update(
            hbm_total_bytes=total,
            resident_delta_bytes=baseline - free,
            memory_scope="HBM free difference after synchronized API with live inputs/outputs; not peak or GPU ratio",
        )
        solver.release()
        # Release internal stream workspace before measuring the next independent case.
        solver.call("aclsolverDestroy", solver.handle)
        solver.handle = ct.c_void_p()
        solver.call("aclsolverCreate", ct.byref(solver.handle))
        solver.call("aclsolverSetStream", solver.handle, solver.stream)
        restored, _ = free_hbm()
        row["resident_delta_after_release_bytes"] = baseline - restored
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(
            json.dumps(
                dict(
                    library=str(args.library),
                    library_sha256=hashlib.sha256(
                        args.library.read_bytes()
                    ).hexdigest(),
                    script_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    cases=len(rows),
                    passed=sum(r["passed"] for r in rows),
                    results=rows,
                ),
                indent=2,
            )
            + "\n"
        )
        if not row["passed"]:
            raise RuntimeError("Isolated memory-test functional failure")

    try:
        for kind, m, n, pad in [
            ("sgetrf", 32, 47, 8),
            ("sgetrf", 32, 600, 0),
            ("sgetrf", 50, 300, 8),
            ("sgetrf", 122, 7940, 8),
            ("sgetrf", 257, 513, 32),
            ("sgetri", 2, 2, 8),
            ("sgetri", 31, 31, 0),
            ("sgetri", 71, 71, 8),
            ("sgetri", 99, 99, 0),
            ("sgetri", 109, 109, 32),
            ("sgetri", 128, 128, 0),
            ("sgetri", 129, 129, 8),
            ("sgetri_nopivot", 129, 129, 8),
            ("sgetri_nopivot_singular", 4096, 4096, 8),
            ("sgetri", 257, 257, 32),
        ]:
            singular = kind == "sgetri_nopivot_singular"
            no_pivot = kind.startswith("sgetri_nopivot")
            kind = "sgetri" if no_pivot else kind
            baseline, _ = free_hbm()
            ld = m + pad
            matrix = np.full((n, ld), -777, np.float32)
            matrix[:, :m] = 0
            diagonal = np.arange(min(m, n))
            matrix[diagonal, diagonal] = 0 if singular else 2
            initial = matrix.copy()
            a = solver.upload(matrix)
            piv = solver.upload(np.arange(1, min(m, n) + 1, dtype=np.int32))
            info = solver.allocate(4)
            size = ct.c_int()
            if kind == "sgetrf":
                solver.call(
                    "aclsolverSgetrf_bufferSize",
                    solver.handle,
                    ct.c_int(m),
                    ct.c_int(n),
                    a,
                    ct.c_int(ld),
                    ct.byref(size),
                )
            else:
                solver.call(
                    "aclsolverSgetri_bufferSize",
                    solver.handle,
                    ct.c_int(n),
                    a,
                    ct.c_int(ld),
                    ct.byref(size),
                )
            work = solver.allocate(size.value * 4) if size.value else ct.c_void_p()
            if kind == "sgetrf":
                solver.call(
                    "aclsolverSgetrf",
                    solver.handle,
                    ct.c_int(m),
                    ct.c_int(n),
                    a,
                    ct.c_int(ld),
                    work,
                    piv,
                    info,
                )
            else:
                solver.call(
                    "aclsolverSgetri",
                    solver.handle,
                    ct.c_int(n),
                    a,
                    ct.c_int(ld),
                    ct.c_void_p() if no_pivot else piv,
                    work,
                    size,
                    info,
                )
            solver.sync()
            solver.download(a, matrix)
            observed = int(solver.download(info, np.empty(1, np.int32))[0])
            if kind == "sgetri" and not singular:
                initial[diagonal, diagonal] = 0.5
            record(
                dict(
                    kind=kind,
                    m=m,
                    n=n,
                    lda=ld,
                    workspace_elements=size.value,
                    pivot=not no_pivot,
                    info=observed,
                    caller_allocated_bytes=matrix.nbytes
                    + min(m, n) * 4
                    + 4
                    + size.value * 4,
                    passed=bool(
                        observed == (1 if singular else 0)
                        and np.array_equal(matrix, initial)
                    ),
                ),
                baseline,
            )
        for kind, n, batch, pad in [
            ("sgetribatched", 2, 129, 0),
            ("sgetribatched", 4, 65537, 0),
            ("sgetribatched", 8, 65537, 0),
            ("sgetribatched", 16, 8193, 0),
            ("sgetribatched", 32, 33, 8),
            ("sgetribatched", 57, 4096, 8),
            ("sgetribatched", 66, 4096, 8),
            ("sgetribatched", 128, 4096, 32),
            ("sgetribatched", 136, 1, 8),
            ("sgetribatched", 256, 1, 0),
            ("smatinvbatched", 1, 8, 8),
            ("smatinvbatched", 2, 129, 0),
            ("smatinvbatched", 3, 128, 8),
            ("smatinvbatched", 3, 32768, 8),
            ("smatinvbatched", 4, 33, 0),
            ("smatinvbatched", 4, 8192, 8),
            ("smatinvbatched", 5, 4097, 8),
            ("smatinvbatched", 10, 1, 8),
            ("smatinvbatched", 10, 4097, 8),
            ("smatinvbatched", 13, 4096, 8),
            ("smatinvbatched", 13, 5376, 8),
            ("smatinvbatched", 16, 1, 8),
            ("smatinvbatched", 16, 4097, 8),
            ("smatinvbatched", 17, 4096, 0),
            ("smatinvbatched", 17, 5376, 0),
            ("smatinvbatched", 18, 4096, 0),
            ("smatinvbatched", 19, 4096, 8),
            ("smatinvbatched", 21, 4096, 0),
            ("smatinvbatched", 22, 4096, 0),
            ("smatinvbatched", 27, 4096, 0),
            ("smatinvbatched", 28, 4096, 8),
            ("smatinvbatched", 29, 16384, 0),
            ("smatinvbatched", 30, 4096, 0),
            ("smatinvbatched", 32, 4096, 0),
            ("smatinvbatched", 32, 33, 8),
        ]:
            baseline, _ = free_hbm()
            ld = n + pad
            source = np.full((batch, n, ld), -777, np.float32)
            source[:, :, :n] = 0
            indices = np.arange(n)
            source[:, indices, indices] = 2
            result = np.full_like(source, -999)
            expected = result.copy()
            expected[:, :, :n] = np.eye(n, dtype=np.float32) * 0.5
            a = solver.upload(source)
            c = solver.upload(result)
            offsets = np.arange(batch, dtype=np.uint64) * n * ld * 4
            ap = solver.upload(np.asarray(a.value + offsets, np.uint64))
            cp = solver.upload(np.asarray(c.value + offsets, np.uint64))
            pivots = np.tile(np.arange(1, n + 1, dtype=np.int32), (batch, 1))
            piv = solver.upload(pivots)
            info = solver.allocate(batch * 4)
            if kind == "smatinvbatched":
                solver.call(
                    "aclsolverSmatinvBatched",
                    solver.handle,
                    ct.c_int(n),
                    ap,
                    ct.c_int(ld),
                    cp,
                    ct.c_int(ld),
                    info,
                    ct.c_int(batch),
                )
            else:
                solver.call(
                    "aclsolverSgetriBatched",
                    solver.handle,
                    ct.c_int(n),
                    ap,
                    ct.c_int(ld),
                    piv,
                    cp,
                    ct.c_int(ld),
                    info,
                    ct.c_int(batch),
                )
            solver.sync()
            solver.download(c, result)
            observed = solver.download(info, np.empty(batch, np.int32))
            preserved = np.array_equal(
                source, solver.download(a, np.empty_like(source))
            )
            pivot_preserved = np.array_equal(
                pivots, solver.download(piv, np.empty_like(pivots))
            )
            record(
                dict(
                    kind=kind,
                    n=n,
                    batch=batch,
                    lda=ld,
                    info_nonzero=int(np.count_nonzero(observed)),
                    input_preserved=bool(preserved),
                    pivots_preserved=bool(pivot_preserved),
                    caller_allocated_bytes=source.nbytes
                    + result.nbytes
                    + 2 * batch * 8
                    + pivots.nbytes
                    + batch * 4,
                    passed=bool(
                        not observed.any()
                        and np.array_equal(result, expected)
                        and preserved
                        and pivot_preserved
                    ),
                ),
                baseline,
            )
    finally:
        solver.close()
    return int(len(rows) != 50 or not all(r["passed"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
