# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check raw-inverse correction and scheduling boundaries with independent leading dimensions."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def check_case(
    solver,
    reference,
    ratio_batch,
    n,
    batch,
    padding,
    seed,
    output_directory,
    matrix_override=None,
):
    lda, ldc = n + padding, n + 2 * padding
    if matrix_override is not None:
        matrix = np.repeat(matrix_override[None], batch, axis=0).astype(np.float32)
    else:
        rng = np.random.default_rng(seed)
        if seed % 2:
            matrix = rng.normal(-0.7, 1.2, (batch, n, n)).astype(np.float32)
        else:
            matrix = rng.uniform(-5, 5, (batch, n, n)).astype(np.float32)
        diagonal = np.arange(n)
        matrix[:, diagonal, diagonal] += np.float32(n)
    source = np.full((batch, n, lda), -777, np.float32)
    source[:, :, :n] = matrix.transpose(0, 2, 1)
    factors = source.copy()
    pivots, factor_info = np.empty((batch, n), np.int32), np.empty(batch, np.int32)
    reference.factor(factors, pivots, factor_info)
    result = np.full((batch, n, ldc), -999, np.float32)
    golden, cpu_info = result.copy(), np.empty(batch, np.int32)
    reference.inverse(factors, pivots, golden, cpu_info)
    input_sha = hashlib.sha256(source).hexdigest()
    a, c = solver.upload(source), solver.upload(result)
    ap = solver.upload(a.value + np.arange(batch, dtype=np.uint64) * n * lda * 4)
    cp = solver.upload(c.value + np.arange(batch, dtype=np.uint64) * n * ldc * 4)
    info = solver.allocate(batch * 4)
    first, deterministic = None, True
    for _ in range(2):
        solver.call(
            "aclsolverSmatinvBatched",
            solver.handle,
            ct.c_int(n),
            ap,
            ct.c_int(lda),
            cp,
            ct.c_int(ldc),
            info,
            ct.c_int(batch),
        )
        solver.sync()
        solver.download(c, result)
        observed_info = solver.download(info, np.empty(batch, np.int32))
        current = hashlib.sha256(result).hexdigest()
        deterministic &= first is None or first == current
        first = current
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        measured = ratio_batch(
            matrix.astype(np.float64), result[:, :, :n].transpose(0, 2, 1)
        )
        baseline = ratio_batch(
            matrix.astype(np.float64), golden[:, :, :n].transpose(0, 2, 1)
        )
    limits = np.maximum(0.5, 5 * baseline)
    nonsingular = cpu_info == 0
    passed = (observed_info == cpu_info) & (
        ~nonsingular | (np.isfinite(measured) & (measured <= limits))
    )
    received = solver.download(a, np.empty_like(source))
    preserved = hashlib.sha256(received).hexdigest() == input_sha
    padding_preserved = bool(np.all(result[:, :, n:] == -999))
    row = dict(
        n=n,
        batch=batch,
        padding=padding,
        lda=lda,
        ldc=ldc,
        seed=seed,
        matrices_passed=int(passed.sum()),
        matrices_total=batch,
        singular_matrices=int(np.count_nonzero(cpu_info)),
        retained_regression=matrix_override is not None,
        max_ratio=float(measured[nonsingular].max()) if nonsingular.any() else None,
        failed_indices=np.flatnonzero(~passed)[:100].tolist(),
        deterministic=bool(deterministic),
        input_preserved=preserved,
        padding_preserved=padding_preserved,
        input_sha256=input_sha,
        output_sha256=hashlib.sha256(result).hexdigest(),
        passed=bool(passed.all() and deterministic and preserved and padding_preserved),
    )
    if not row["passed"]:
        bad = np.flatnonzero(~passed)[:32]
        if not len(bad):
            bad = np.arange(min(32, batch))
        np.savez(
            output_directory / f"raw{n}_b{batch}_p{padding}_failure.npz",
            matrix_indices=bad,
            input=source[bad],
            actual=result[bad],
            cpu=golden[bad],
            info=observed_info[bad],
            cpu_info=cpu_info[bad],
        )
    solver.release()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.tests_root))
    from validate import Solver
    from validate_batch import Reference, ratio_batch

    solver, reference = Solver(args.library, 0), Reference(args.reference_bridge)
    records = []
    batches = (4095, 4096, 4097, 8191, 8192, 8193, 32767, 32768, 32769)
    try:
        with (args.output / "results.jsonl").open("x") as stream:
            for n in (3, 4, 5, *range(10, 33)):
                case_batches = (
                    (511, 512, 513, 4095, 4096, 4097)
                    if n >= 17
                    else batches + ((5375, 5376, 5377) if 10 <= n <= 13 else ())
                )
                for batch in case_batches:
                    for padding in (0, 8):
                        row = check_case(
                            solver,
                            reference,
                            ratio_batch,
                            n,
                            batch,
                            padding,
                            20261016 + n * 100000 + batch,
                            args.output,
                        )
                        records.append(row)
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        print(json.dumps(row), flush=True)
                        if not row["passed"]:
                            raise RuntimeError(
                                "Raw-inverse boundary failure; see retained inputs"
                            )
            fixture = json.loads(
                (args.tests_root / "cases/regression_raw10_status.json").read_text()
            )
            words = np.asarray(
                [int(value, 16) for value in fixture["words_hex"]], dtype="<u4"
            )
            assert hashlib.sha256(words).hexdigest() == fixture["input_sha256"]
            matrix = words.view("<f4").reshape(10, 10).T.copy()
            for batch in (1, 33, 4097):
                for padding in (0, 8):
                    row = check_case(
                        solver,
                        reference,
                        ratio_batch,
                        10,
                        batch,
                        padding,
                        fixture["generation_seed"],
                        args.output,
                        matrix,
                    )
                    records.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(json.dumps(row), flush=True)
                    if not row["passed"]:
                        raise RuntimeError("Retained raw10 status regression failed")
    finally:
        solver.close()
        summary = dict(
            library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
            reference_sha256=hashlib.sha256(reference.path.read_bytes()).hexdigest(),
            bridge_sha256=hashlib.sha256(
                args.reference_bridge.read_bytes()
            ).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            tests=len(records),
            passed=sum(r["passed"] for r in records),
            matrices_total=sum(r["matrices_total"] for r in records),
            failed=[r for r in records if not r["passed"]],
            results=records,
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert summary["tests"] == summary["passed"] == 402
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
