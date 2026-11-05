# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Additional no-diagonal-shift, singular, integer and strided-pointer Device tests."""

import argparse
import ctypes as ct
import gzip
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
    import validate as single
    from validate_batch import Reference, ratio_batch

    args.output.mkdir(parents=True, exist_ok=False)
    solver = single.Solver(args.library, 0)
    reference = Reference(args.reference_bridge)
    rows = []

    def record(row):
        rows.append(row)
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        solver.release()
        if not row["passed"]:
            raise RuntimeError(f"Isolated supplemental failure {len(rows) - 1}")

    def batch_case(kind, n, batch, pad, seed, mode="random", stride=1, pivot=True):
        rng = np.random.default_rng(seed)
        mu, sigma = float(rng.uniform(-5, 5)), float(rng.uniform(0.1, 2))
        normal = seed % 2 == 1
        shape = (batch, n, n)
        values = (
            rng.normal(mu, sigma, shape) if normal else rng.uniform(-5, 5, shape)
        ).astype(np.float32)
        if mode == "singular":
            values[:, -1, :] = 0
        elif mode == "near":
            values[:] = np.eye(n, dtype=np.float32)
            values[:, 0, 1] = values[:, 1, 0] = 1
            values[:, 1, 1] = np.float32(1 + 2**-20)
        elif mode == "integer":
            values[:] = rng.integers(-5, 6, shape)
        ld = n + pad
        source = np.full((batch, n, ld), -777, np.float32)
        source[:, :, :n] = values
        factors = source.copy()
        pivots = np.empty((batch, n), np.int32)
        factor_info = np.empty(batch, np.int32)
        if pivot or kind == "smatinvbatched":
            reference.factor(factors, pivots, factor_info)
        else:
            for i in range(batch):
                factor, _, info = single.factor_reference(source[i, :, :n].T, False)
                factors[i, :, :n] = factor.T
                pivots[i] = np.arange(1, n + 1, dtype=np.int32)
                factor_info[i] = info
        actual_input = source if kind == "smatinvbatched" else factors
        input_backing = np.full((batch * stride, n, ld), -777, np.float32)
        input_backing[::stride] = actual_input
        output_backing = np.full_like(input_backing, -999)
        matrix = solver.upload(input_backing)
        output = solver.upload(output_backing)
        pointer_step = n * ld * stride * 4
        ap = solver.upload(
            np.asarray(
                matrix.value + np.arange(batch, dtype=np.uint64) * pointer_step,
                np.uint64,
            )
        )
        cp = solver.upload(
            np.asarray(
                output.value + np.arange(batch, dtype=np.uint64) * pointer_step,
                np.uint64,
            )
        )
        pivot_device = solver.upload(pivots)
        infos = solver.allocate(batch * 4)
        first = None
        deterministic = True
        for repeat in range(2):
            if kind == "smatinvbatched":
                solver.call(
                    "aclsolverSmatinvBatched",
                    solver.handle,
                    ct.c_int(n),
                    ap,
                    ct.c_int(ld),
                    cp,
                    ct.c_int(ld),
                    infos,
                    ct.c_int(batch),
                )
            else:
                solver.call(
                    "aclsolverSgetriBatched",
                    solver.handle,
                    ct.c_int(n),
                    ap,
                    ct.c_int(ld),
                    pivot_device if pivot else ct.c_void_p(),
                    cp,
                    ct.c_int(ld),
                    infos,
                    ct.c_int(batch),
                )
            solver.sync()
            solver.download(output, output_backing)
            observed = solver.download(infos, np.empty(batch, np.int32))
            fingerprint = (
                hashlib.sha256(output_backing).hexdigest(),
                observed.tobytes(),
            )
            if first is None:
                first = fingerprint
            deterministic &= fingerprint == first
        result = output_backing[::stride, :, :n]
        cpu = np.empty((batch, n, ld), np.float32)
        cpu_info = np.empty(batch, np.int32)
        reference.inverse(factors, pivots, cpu, cpu_info)
        a = source[:, :, :n].transpose(0, 2, 1).astype(np.float64)
        if kind == "sgetribatched":
            lu = factors[:, :, :n].transpose(0, 2, 1).astype(np.float64)
            a = (np.tril(lu, -1) + np.eye(n)) @ np.triu(lu)
            indices = np.arange(batch)
            for k in range(n - 1, -1, -1):
                targets = pivots[:, k] - 1
                saved = a[:, k].copy()
                a[:, k] = a[indices, targets]
                a[indices, targets] = saved
        with np.errstate(all="ignore"):
            measured = ratio_batch(a, result.transpose(0, 2, 1))
            baseline = ratio_batch(a, cpu[:, :, :n].transpose(0, 2, 1))
        limit = np.maximum(5 * baseline, 0.5 if kind == "smatinvbatched" else 0.05)
        nonsingular = cpu_info == 0
        passed = (observed == cpu_info) & (
            ~nonsingular | (np.isfinite(measured) & (measured <= limit))
        )
        received = solver.download(matrix, np.empty_like(input_backing))
        input_preserved = np.array_equal(received, input_backing)
        padding_preserved = np.all(output_backing[::stride, :, n:] == -999)
        gaps_preserved = all(
            np.all(output_backing[k::stride] == -999) for k in range(1, stride)
        )
        actual_pivots = solver.download(pivot_device, np.empty_like(pivots))
        pivots_preserved = np.array_equal(pivots, actual_pivots)
        label = kind + "_" + str(seed)
        with gzip.open(args.output / (label + ".csv.gz"), "wt") as stream:
            stream.write("matrix,ratio,cpu_ratio,limit,info,cpu_info,passed\n")
            for i in range(batch):
                stream.write(
                    ",".join(
                        map(
                            str,
                            [
                                i,
                                measured[i],
                                baseline[i],
                                limit[i],
                                observed[i],
                                cpu_info[i],
                                passed[i],
                            ],
                        )
                    )
                    + "\n"
                )
        bad = np.flatnonzero(~passed)
        if len(bad):
            np.savez(
                args.output / (label + "_failed.npz"),
                matrix_indices=bad[:32],
                input=actual_input[bad[:32]],
                output=result[bad[:32]],
                cpu=cpu[bad[:32]],
                pivots=pivots[bad[:32]],
            )
        return dict(
            kind=kind,
            n=n,
            batch=batch,
            lda=ld,
            pointer_stride=stride,
            pivot=pivot,
            mode=mode,
            seed=seed,
            diagonal_shift=0,
            distribution="normal" if normal else "uniform",
            mu=mu,
            sigma=sigma,
            matrices_passed=int(np.count_nonzero(passed)),
            matrices_total=batch,
            singular_matrices=int(np.count_nonzero(~nonsingular)),
            failed_indices=bad[:100].tolist(),
            input_preserved=bool(input_preserved),
            padding_preserved=bool(padding_preserved),
            gaps_preserved=bool(gaps_preserved),
            pivots_preserved=bool(pivots_preserved),
            deterministic=bool(deterministic),
            max_ratio=float(np.max(measured[nonsingular]))
            if nonsingular.any()
            else None,
            passed=bool(
                passed.all()
                and deterministic
                and input_preserved
                and padding_preserved
                and gaps_preserved
                and pivots_preserved
            ),
        )

    try:
        # 15 + 20 + 15 + 22 = 72 extra cases without a diagonal shift.
        dimensions = [1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 257]
        for kind, sizes in [
            ("sgetrf", dimensions),
            (
                "sgetri",
                [
                    1,
                    2,
                    3,
                    4,
                    7,
                    8,
                    15,
                    16,
                    31,
                    32,
                    63,
                    64,
                    127,
                    128,
                    129,
                    150,
                    255,
                    256,
                    257,
                    511,
                ],
            ),
        ]:
            for i, n in enumerate(sizes):
                mu, sigma = (-4.5 + 0.4 * (i % 20)), (0.1 + 0.1 * (i % 19))

                def make_matrix(
                    m, columns, rng, distribution, mean=mu, deviation=sigma
                ):
                    v = (
                        rng.uniform(-5, 5, (m, columns))
                        if distribution == "uniform"
                        else rng.normal(mean, deviation, (m, columns))
                    )
                    return np.array(v, dtype=np.float32, order="F")

                single.make_matrix = make_matrix
                columns = 257 if kind == "sgetrf" and n in (8, 16, 32, 64, 128) else n
                case = dict(
                    m=n, n=columns, lda=n + (8 if i % 2 else 32), pivot=i % 3 != 1
                )
                row = single.run_single(solver, kind, case, 1000 + i, 0, 2, args.output)
                row.update(
                    diagonal_shift=0,
                    mu=mu,
                    sigma=sigma,
                    supplemental="no_diagonal_shift",
                )
                record(row)
        for i, n in enumerate(
            [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129]
        ):
            batch = [1, 8, 128, 1024, 3000][i % 5] if n <= 32 else [1, 8][i % 2]
            record(
                batch_case(
                    "sgetribatched",
                    n,
                    batch,
                    8 if i % 2 else 32,
                    3001 + i,
                    pivot=not (i == 1),
                )
            )
        for i in range(22):
            n = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32][i % 10]
            record(
                batch_case(
                    "smatinvbatched",
                    n,
                    [1, 8, 128, 1024, 3000][i % 5],
                    8 if i % 2 else 32,
                    4000 + i,
                )
            )
        for i, (kind, n, batch, mode, stride, pad) in enumerate(
            [
                ("smatinvbatched", 2, 128, "singular", 1, 8),
                ("smatinvbatched", 3, 128, "singular", 2, 32),
                ("sgetribatched", 16, 8, "singular", 2, 8),
                ("sgetribatched", 129, 1, "singular", 1, 32),
                ("smatinvbatched", 2, 128, "near", 2, 8),
                ("smatinvbatched", 3, 128, "near", 1, 32),
                ("smatinvbatched", 2, 128, "integer", 1, 0),
                ("sgetribatched", 4, 128, "integer", 2, 8),
                ("sgetribatched", 4, 65537, "random", 2, 0),
                ("sgetribatched", 8, 65537, "random", 2, 0),
                ("sgetribatched", 16, 8193, "random", 2, 0),
            ]
        ):
            record(batch_case(kind, n, batch, pad, 5000 + i, mode, stride))
    finally:
        solver.close()
        summary = dict(
            library=str(args.library),
            library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
            reference_library=str(reference.path),
            reference_sha256=hashlib.sha256(reference.path.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            tests=len(rows),
            passed=sum(r["passed"] for r in rows),
            failed=[i for i, r in enumerate(rows) if not r["passed"]],
            required_no_diagonal_shift_cases=72,
            results=rows,
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(len(rows) != 83 or not all(r["passed"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
