# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate each batch member through the public NPU API and actual FP32 LAPACK."""

import argparse
import csv
import ctypes as ct
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import Solver  # noqa: E402


def pointer(array):
    return ct.c_void_p(array.ctypes.data)


class Reference:
    def __init__(self, bridge_path):
        site = Path(scipy.__file__).parent.parent
        self.path = next((site / "scipy.libs").glob("*openblas*.so"))
        self.blas = ct.CDLL(str(self.path))
        self.bridge = ct.CDLL(str(bridge_path))
        self.factor_function = ct.cast(self.blas.scipy_sgetrf_, ct.c_void_p)
        self.inverse_function = ct.cast(self.blas.scipy_sgetri_, ct.c_void_p)

    def factor(self, factors, pivots, infos):
        batch, n, ld = factors.shape
        status = self.bridge.lu950_reference_factor(
            ct.c_int(n),
            ct.c_int(batch),
            pointer(factors),
            ct.c_int(ld),
            pointer(pivots),
            pointer(infos),
            self.factor_function,
        )
        if status:
            raise RuntimeError("CPU reference factor bridge failed")

    def inverse(self, factors, pivots, output, infos):
        batch, n, lda = factors.shape
        status = self.bridge.lu950_reference_inverse(
            ct.c_int(n),
            ct.c_int(batch),
            pointer(factors),
            ct.c_int(lda),
            pointer(pivots),
            pointer(output),
            ct.c_int(output.shape[2]),
            pointer(infos),
            self.inverse_function,
        )
        if status:
            raise RuntimeError("CPU reference inverse bridge failed")


def norm1_batch(matrices):
    return np.abs(matrices).sum(axis=1).max(axis=1)


def ratio_batch(a, x):
    x = x.astype(np.float64)
    n = a.shape[1]
    residual = norm1_batch(np.eye(n) - x @ a)
    return residual / (n * norm1_batch(a) * norm1_batch(x) * 2.0**-24)


def run_case(
    solver,
    reference,
    kind,
    case,
    index,
    output_directory,
    warmup,
    samples,
    batch_override=None,
    record_windows=False,
):
    n, batch, lda = case["n"], case["batch"], case["lda"]
    if batch_override is not None:
        batch = min(batch, batch_override)
    ldc = case["lda_inv"] if kind == "smatinvbatched" else case["ldc"]
    rng = np.random.default_rng(20261016 + index)
    source = np.full((batch, n, lda), -777, np.float32)
    distribution = "uniform" if index % 2 == 0 else "normal"
    # Bound temporary generation memory independently of the specified batch size.
    chunk = max(1, 8 * 1024 * 1024 // (n * n))
    for start in range(0, batch, chunk):
        stop = min(start + chunk, batch)
        shape = (stop - start, n, n)
        values = (
            rng.uniform(-5, 5, shape)
            if distribution == "uniform"
            else rng.normal(-0.7, 1.2, shape)
        )
        source[start:stop, :, :n] = values.astype(np.float32)
        diagonal = np.arange(n)
        source[start:stop, diagonal, diagonal] += np.float32(n)
    factors = source.copy()
    pivots = np.empty((batch, n), np.int32)
    cpu_infos = np.empty(batch, np.int32)
    reference.factor(factors, pivots, cpu_infos)
    actual_input = source if kind == "smatinvbatched" else factors
    input_digest = hashlib.sha256(actual_input).hexdigest()
    matrix = solver.upload(actual_input)
    result = np.full((batch, n, ldc), -999, np.float32)
    inverse = solver.upload(result)
    pivot_buffer = solver.upload(pivots)
    infos = solver.allocate(batch * 4)
    ap = solver.upload(
        np.asarray(
            matrix.value + np.arange(batch, dtype=np.uint64) * n * lda * 4,
            dtype=np.uint64,
        )
    )
    cp = solver.upload(
        np.asarray(
            inverse.value + np.arange(batch, dtype=np.uint64) * n * ldc * 4,
            dtype=np.uint64,
        )
    )
    first_digest = None
    deterministic = True
    initial_info = None
    count = max(2, warmup + samples)
    call_windows = []
    for repeat in range(count):
        begin = time.time_ns() if record_windows else None
        if kind == "smatinvbatched":
            solver.call(
                "aclsolverSmatinvBatched",
                solver.handle,
                ct.c_int(n),
                ap,
                ct.c_int(lda),
                cp,
                ct.c_int(ldc),
                infos,
                ct.c_int(batch),
            )
        else:
            solver.call(
                "aclsolverSgetriBatched",
                solver.handle,
                ct.c_int(n),
                ap,
                ct.c_int(lda),
                pivot_buffer,
                cp,
                ct.c_int(ldc),
                infos,
                ct.c_int(batch),
            )
        solver.sync()
        if record_windows:
            call_windows.append(
                dict(
                    iteration=repeat,
                    warmup=repeat < warmup,
                    begin_utc_ns=begin,
                    end_utc_ns=time.time_ns(),
                )
            )
        if repeat in (0, count - 1):
            solver.download(inverse, result)
            observed_info = solver.download(infos, np.empty(batch, np.int32))
            digest = hashlib.sha256(result).hexdigest()
            if repeat == 0:
                first_digest, initial_info = digest, observed_info.copy()
            else:
                deterministic = digest == first_digest and np.array_equal(
                    observed_info, initial_info
                )
    received_input = np.empty_like(actual_input)
    solver.download(matrix, received_input)
    input_preserved = hashlib.sha256(received_input).hexdigest() == input_digest
    del received_input
    received_pivots = solver.download(pivot_buffer, np.empty_like(pivots))
    pivots_preserved = bool(np.array_equal(received_pivots, pivots))
    del received_pivots
    padding_preserved = bool(np.all(result[:, :, n:] == -999))
    rows_path = output_directory / f"{kind}_{index}_matrices.csv.gz"
    maximum_ratio, failed, failed_indices = 0.0, 0, []
    # FP64 residuals are evaluated in bounded chunks; all matrices get a row.
    validation_chunk = max(1, 16 * 1024 * 1024 // (n * n * 8))
    with gzip.open(rows_path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["matrix", "ratio", "cpu_ratio", "limit", "info", "cpu_info", "passed"]
        )
        for start in range(0, batch, validation_chunk):
            stop = min(start + validation_chunk, batch)
            cpu = np.empty((stop - start, n, ldc), np.float32)
            cpu_info = np.empty(stop - start, np.int32)
            reference.inverse(factors[start:stop], pivots[start:stop], cpu, cpu_info)
            if kind == "smatinvbatched":
                a = source[start:stop, :, :n].transpose(0, 2, 1).astype(np.float64)
                floor = 0.5
            else:
                lu = factors[start:stop, :, :n].transpose(0, 2, 1).astype(np.float64)
                a = (np.tril(lu, -1) + np.eye(n)) @ np.triu(lu)
                local_pivots = pivots[start:stop] - 1
                matrix_indices = np.arange(stop - start)
                for k in range(n - 1, -1, -1):
                    saved = a[:, k].copy()
                    a[:, k] = a[matrix_indices, local_pivots[:, k]]
                    a[matrix_indices, local_pivots[:, k]] = saved
                floor = 0.05
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                measured = ratio_batch(a, result[start:stop, :, :n].transpose(0, 2, 1))
                baseline = ratio_batch(a, cpu[:, :, :n].transpose(0, 2, 1))
            limits = np.maximum(5 * baseline, floor)
            nonsingular = cpu_info == 0
            passed = (observed_info[start:stop] == cpu_info) & (
                ~nonsingular | (np.isfinite(measured) & (measured <= limits))
            )
            if nonsingular.any():
                maximum_ratio = max(maximum_ratio, float(np.max(measured[nonsingular])))
            measured[~nonsingular] = np.nan
            baseline[~nonsingular] = np.nan
            limits[~nonsingular] = np.nan
            failed += int(np.count_nonzero(~passed))
            failed_indices.extend(
                (
                    np.flatnonzero(~passed)[: max(0, 100 - len(failed_indices))] + start
                ).tolist()
            )
            writer.writerows(
                zip(
                    range(start, stop),
                    measured,
                    baseline,
                    limits,
                    observed_info[start:stop],
                    cpu_info,
                    passed,
                    strict=True,
                )
            )
            if not passed.all():
                bad = np.flatnonzero(~passed)[:32]
                np.savez(
                    output_directory / f"{kind}_{index}_failed_chunk{start}.npz",
                    matrix_indices=bad + start,
                    input=actual_input[start:stop][bad],
                    output=result[start:stop][bad],
                    cpu=cpu[bad],
                    pivots=pivots[start:stop][bad],
                )
    return dict(
        kind=kind,
        case_index=index,
        parameters=case,
        actual_batch=batch,
        task_batch_covered=batch == case["batch"],
        distribution=distribution,
        input_sha256=input_digest,
        output_sha256=hashlib.sha256(result).hexdigest(),
        matrices_total=batch,
        singular_matrices=int(np.count_nonzero(cpu_infos)),
        matrices_passed=batch - failed,
        failed_indices=failed_indices,
        failed_indices_truncated=failed > len(failed_indices),
        max_ratio=maximum_ratio,
        deterministic=bool(deterministic),
        input_preserved=input_preserved,
        pivots_preserved=pivots_preserved,
        padding_preserved=padding_preserved,
        matrix_csv=str(rows_path),
        call_windows=call_windows,
        window_scope="Host timestamps associate msprof kernels only; never used as device durations",
        passed=bool(
            failed == 0
            and deterministic
            and input_preserved
            and pivots_preserved
            and padding_preserved
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-bridge", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=("sgetribatched", "smatinvbatched"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--batch-override", type=int)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--record-windows", action="store_true")
    parser.add_argument("--stream-per-case", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    indices = (
        list(range(len(cases)))
        if args.indices == "all"
        else [int(v) for v in args.indices.split(",")]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = args.output.parent / ("case_" + args.output.stem)
    evidence.mkdir(exist_ok=True)
    solver, reference = (
        Solver(args.library, args.device),
        Reference(args.reference_bridge),
    )
    failures, executed = [], 0
    try:
        with args.output.open("x") as stream:
            environment = dict(
                numpy=np.__version__,
                scipy=scipy.__version__,
                scipy_file=scipy.__file__,
                library=str(args.library.resolve()),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                reference_library=str(reference.path.resolve()),
                reference_sha256=hashlib.sha256(
                    reference.path.read_bytes()
                ).hexdigest(),
                reference_symbols=["scipy_sgetrf_", "scipy_sgetri_"],
                bridge_sha256=hashlib.sha256(
                    args.reference_bridge.read_bytes()
                ).hexdigest(),
            )
            stream.write(json.dumps(dict(environment=environment)) + "\n")
            stream.flush()
            for index in indices:
                stream_id = solver.use_unique_stream() if args.stream_per_case else None
                try:
                    result = run_case(
                        solver,
                        reference,
                        args.kind,
                        cases[index],
                        index,
                        evidence,
                        args.warmup,
                        args.samples,
                        args.batch_override,
                        args.record_windows,
                    )
                except Exception as error:
                    result = dict(
                        kind=args.kind,
                        case_index=index,
                        parameters=cases[index],
                        passed=False,
                        error=repr(error),
                    )
                if stream_id is not None:
                    result["device_stream_id"] = stream_id
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(json.dumps(result), flush=True)
                if not result["passed"]:
                    failures.append(index)
                executed += 1
                solver.release()
                if failures and args.stop_on_failure:
                    break
    finally:
        solver.close()
    print(
        json.dumps(
            dict(planned_cases=len(indices), executed_cases=executed, failures=failures)
        ),
        flush=True,
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
