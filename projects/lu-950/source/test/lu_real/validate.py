# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the public Device API with FP32 LAPACK and FP64 residuals.

Run with a private NumPy/SciPy environment on an authorized Ascend 950 device.
The JSONL output distinguishes exact source identity, functional validation and
kernel timing. This program never substitutes a CPU result for an NPU output.
"""

import argparse
import ctypes as ct
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.linalg.lapack import sgetrf, sgetri


class Solver:
    def __init__(self, library, device):
        self.runtime = ct.CDLL("libascendcl.so", mode=ct.RTLD_GLOBAL)
        self.library = ct.CDLL(str(library), mode=ct.RTLD_GLOBAL)
        self.allocations = []
        self.parked_streams = []
        self.stream, self.handle = ct.c_void_p(), ct.c_void_p()
        self.device = device
        self.call("aclInit", ct.c_void_p(), runtime=True)
        self.call("aclrtSetDevice", ct.c_int(device), runtime=True)
        self.call("aclrtCreateStream", ct.byref(self.stream), runtime=True)
        self.call("aclsolverCreate", ct.byref(self.handle))
        self.call("aclsolverSetStream", self.handle, self.stream)
        self.sync()

    def call(self, name, *args, runtime=False, expected=0):
        result = getattr(self.runtime if runtime else self.library, name)(*args)
        if result != expected:
            raise RuntimeError(f"{name}: {result}, expected {expected}")
        return result

    def use_unique_stream(self):
        self.sync()
        previous = self.stream
        stream = ct.c_void_p()
        self.call("aclrtCreateStream", ct.byref(stream), runtime=True)
        try:
            self.call("aclsolverSetStream", self.handle, stream)
        except Exception:
            self.call("aclrtDestroyStream", stream, runtime=True)
            raise
        self.parked_streams.append(previous)
        self.stream = stream
        identifier = ct.c_int32()
        self.call("aclrtStreamGetId", stream, ct.byref(identifier), runtime=True)
        return identifier.value

    def allocate(self, size):
        pointer = ct.c_void_p()
        self.call(
            "aclrtMalloc",
            ct.byref(pointer),
            ct.c_size_t(max(size, 32)),
            ct.c_int(0),
            runtime=True,
        )
        self.allocations.append(pointer)
        return pointer

    def upload(self, array):
        assert array.flags.c_contiguous or array.flags.f_contiguous
        pointer = self.allocate(array.nbytes)
        self.write(pointer, array)
        return pointer

    def write(self, pointer, array):
        self.call(
            "aclrtMemcpy",
            pointer,
            ct.c_size_t(array.nbytes),
            ct.c_void_p(array.ctypes.data),
            ct.c_size_t(array.nbytes),
            ct.c_int(1),
            runtime=True,
        )

    def download(self, pointer, array):
        self.call(
            "aclrtMemcpy",
            ct.c_void_p(array.ctypes.data),
            ct.c_size_t(array.nbytes),
            pointer,
            ct.c_size_t(array.nbytes),
            ct.c_int(2),
            runtime=True,
        )
        return array

    def sync(self):
        self.call("aclrtSynchronizeStream", self.stream, runtime=True)

    def release(self):
        self.sync()
        for pointer in reversed(self.allocations):
            self.call("aclrtFree", pointer, runtime=True)
        self.allocations.clear()

    def close(self):
        self.release()
        self.call("aclsolverDestroy", self.handle)
        self.call("aclrtDestroyStream", self.stream, runtime=True)
        for stream in self.parked_streams:
            self.call("aclrtDestroyStream", stream, runtime=True)
        self.parked_streams.clear()
        self.call("aclrtResetDevice", ct.c_int(self.device), runtime=True)
        self.call("aclFinalize", runtime=True)


def norm1(a):
    return float(np.abs(a).sum(axis=0).max())


def inverse_ratio(a, inverse):
    x = inverse.astype(np.float64)
    return norm1(np.eye(len(a)) - x @ a) / (len(a) * norm1(a) * norm1(x) * 2.0**-24)


def factors_matrix(lu, pivots):
    m, n = lu.shape
    rank = min(m, n)
    lower = np.tril(lu[:, :rank].astype(np.float64), -1) + np.eye(m, rank)
    upper = np.triu(lu[:rank].astype(np.float64))
    matrix = lower @ upper
    if pivots is not None:
        for k in range(rank - 1, -1, -1):
            matrix[[k, pivots[k]]] = matrix[[pivots[k], k]]
    return matrix


def make_matrix(m, n, rng, distribution):
    values = (
        rng.uniform(-5, 5, (m, n))
        if distribution == "uniform"
        else rng.normal(-0.7, 1.2, (m, n))
    )
    values = np.array(values, dtype=np.float32, order="F")
    indices = np.arange(min(m, n))
    values[indices, indices] += np.float32(n)
    return values


def factor_reference(a, pivot):
    factors, pivots, info = sgetrf(np.array(a, dtype=np.float32, order="F"))
    if pivot or np.array_equal(pivots, np.arange(min(a.shape))):
        return factors, pivots, info
    factors = a.copy(order="F")
    info = 0
    for k in range(min(a.shape)):
        if factors[k, k] == 0:
            info = info or k + 1
            continue
        factors[k + 1 :, k] /= factors[k, k]
        factors[k + 1 :, k + 1 :] -= (
            factors[k + 1 :, k, None] * factors[None, k, k + 1 :]
        )
    return factors, np.arange(min(a.shape), dtype=np.int32), info


def run_single(
    solver, kind, case, index, warmup, samples, evidence=None, record_windows=False
):
    n, m = case["n"], case.get("m", case["n"])
    ld = case["lda"]
    pivot = case.get("pivot", True)
    distribution = "uniform" if index % 2 == 0 else "normal"
    original = make_matrix(m, n, np.random.default_rng(20261016 + index), distribution)
    cpu_lu, cpu_pivots, cpu_info = factor_reference(original, pivot)
    packed = np.full((ld, n), -777, dtype=np.float32, order="F")
    packed[:m] = original if kind == "sgetrf" else cpu_lu
    initial = packed.copy(order="F")
    matrix = solver.upload(packed)
    pivots = solver.upload(np.asarray(cpu_pivots + 1, dtype=np.int32))
    info = solver.allocate(4)
    workspace_elements = ct.c_int()
    query_args = (ct.c_int(m), ct.c_int(n)) if kind == "sgetrf" else (ct.c_int(n),)
    solver.call(
        "aclsolver" + kind[0].upper() + kind[1:] + "_bufferSize",
        solver.handle,
        *query_args,
        matrix,
        ct.c_int(ld),
        ct.byref(workspace_elements),
    )
    workspace = (
        solver.allocate(workspace_elements.value * 4)
        if workspace_elements.value
        else ct.c_void_p()
    )
    function = "aclsolver" + kind[0].upper() + kind[1:]
    pivot_argument = pivots if pivot else ct.c_void_p()

    def invoke():
        if kind == "sgetrf":
            solver.call(
                function,
                solver.handle,
                ct.c_int(m),
                ct.c_int(n),
                matrix,
                ct.c_int(ld),
                workspace,
                pivot_argument,
                info,
            )
        else:
            solver.call(
                function,
                solver.handle,
                ct.c_int(n),
                matrix,
                ct.c_int(ld),
                pivot_argument,
                workspace,
                ct.c_int(workspace_elements.value),
                info,
            )

    first_output = None
    differing_elements = []
    mismatch_output = None
    mismatch_iteration = None
    first_pivots = None
    deterministic = True
    call_windows = []
    for iteration in range(max(2, warmup + samples)):
        solver.write(matrix, initial)
        begin = time.time_ns() if record_windows else None
        invoke()
        solver.sync()
        if record_windows:
            call_windows.append(
                dict(
                    iteration=iteration,
                    warmup=iteration < warmup,
                    begin_utc_ns=begin,
                    end_utc_ns=time.time_ns(),
                )
            )
        solver.download(matrix, packed)
        current_pivots = solver.download(pivots, np.empty(min(m, n), np.int32)).copy()
        current_info = solver.download(info, np.empty(1, np.int32))[0]
        if iteration == 0:
            first_output, first_pivots, first_info = (
                packed.copy(order="F"),
                current_pivots,
                current_info,
            )
        same = (
            packed.tobytes(order="F") == first_output.tobytes(order="F")
            and np.array_equal(current_pivots, first_pivots)
            and current_info == first_info
        )
        deterministic &= same
        if not same and mismatch_output is None:
            mismatch_output = packed.copy(order="F")
            mismatch_iteration = iteration
    if not deterministic:
        positions = np.argwhere(
            mismatch_output.view(np.uint32) != first_output.view(np.uint32)
        )
        differing_elements = [
            dict(
                row=int(row),
                column=int(col),
                first=float(first_output[row, col]),
                last=float(mismatch_output[row, col]),
            )
            for row, col in positions[:16]
        ]
    actual = packed[:m]
    cpu = cpu_lu if kind == "sgetrf" else sgetri(cpu_lu.copy(order="F"), cpu_pivots)[0]
    if kind == "sgetrf":
        denominator = n * norm1(original.astype(np.float64)) * 2.0**-24
        reconstructed = factors_matrix(actual, first_pivots - 1 if pivot else None)
        measured_ratio = norm1(reconstructed - original) / denominator
        cpu_ratio = (
            norm1(factors_matrix(cpu_lu, cpu_pivots if pivot else None) - original)
            / denominator
        )
        floor = 0.3
    else:
        a = factors_matrix(cpu_lu, cpu_pivots if pivot else None)
        measured_ratio, cpu_ratio = inverse_ratio(a, actual), inverse_ratio(a, cpu)
        floor = 0.05
    limit = max(5 * cpu_ratio, floor)
    padding_ok = bool(np.all(packed[m:] == -777))
    pivots_preserved = kind != "sgetri" or np.array_equal(first_pivots, cpu_pivots + 1)
    passed = bool(
        deterministic
        and padding_ok
        and pivots_preserved
        and current_info == cpu_info
        and np.isfinite(measured_ratio)
        and measured_ratio <= limit
    )
    if not passed and evidence is not None:
        failed = evidence / (kind + "_failed_" + str(index) + ".npz")
        np.savez(
            failed,
            initial=initial,
            output=packed,
            first_output=first_output,
            pivots=first_pivots,
            mismatch_output=mismatch_output,
            cpu=cpu,
        )
    return dict(
        kind=kind,
        case_index=index,
        parameters=case,
        distribution=distribution,
        differing_elements=differing_elements,
        mismatch_iteration=mismatch_iteration,
        source_input_sha256=hashlib.sha256(initial.tobytes(order="F")).hexdigest(),
        ratio=measured_ratio,
        cpu_ratio=cpu_ratio,
        ratio_limit=limit,
        info=int(current_info),
        cpu_info=int(cpu_info),
        deterministic=bool(deterministic),
        padding_preserved=padding_ok,
        pivots_preserved=bool(pivots_preserved),
        passed=passed,
        output_sha256=hashlib.sha256(packed.tobytes(order="F")).hexdigest(),
        workspace_elements=workspace_elements.value,
        call_windows=call_windows,
        window_scope="Host timestamps associate msprof kernels only; never used as device durations",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--kind", choices=("sgetrf", "sgetri"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--record-windows", action="store_true")
    parser.add_argument("--stream-per-case", action="store_true")
    args = parser.parse_args()
    assert args.library.is_file() and args.cases.is_file()
    cases = json.loads(args.cases.read_text())["cases"]
    indices = (
        list(range(len(cases)))
        if args.indices == "all"
        else [int(value) for value in args.indices.split(",")]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = args.output.parent / ("case_" + args.output.stem)
    evidence.mkdir(exist_ok=True)
    solver = Solver(args.library, args.device)
    failures = []
    try:
        with args.output.open("x") as output:
            environment = dict(
                numpy=np.__version__,
                numpy_file=np.__file__,
                library=str(args.library.resolve()),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
            )
            output.write(json.dumps(dict(environment=environment)) + "\n")
            output.flush()
            for index in indices:
                stream_id = solver.use_unique_stream() if args.stream_per_case else None
                try:
                    result = run_single(
                        solver,
                        args.kind,
                        cases[index],
                        index,
                        args.warmup,
                        args.samples,
                        evidence,
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
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(json.dumps(result), flush=True)
                if not result["passed"]:
                    failures.append(index)
                solver.release()
    finally:
        solver.close()
    print(json.dumps(dict(cases=len(indices), failures=failures)), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
