# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Analyze full saved batch tensors in bounded memory; retain every matrix result."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_batch import Reference, ratio_batch  # noqa: E402


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def analyze(case, reference, baselines):
    meta = json.loads((case / "batch_getri_application.json").read_text())
    n, batch, ld = meta["n"], meta["batch"], meta["n"] + meta["padding"]
    factors = np.memmap(
        case / "input.bin", dtype=np.float32, mode="r", shape=(batch, n, ld)
    )
    outputs = np.memmap(
        case / "output.bin", dtype=np.float32, mode="r", shape=(batch, n, ld)
    )
    pivots = np.memmap(case / "pivots.bin", dtype=np.int32, mode="r", shape=(batch, n))
    infos = np.memmap(case / "info.bin", dtype=np.int32, mode="r", shape=(batch,))
    current_numbering = meta.get("argument_numbering") == "excludes_handle"
    chunk = max(1, 16 * 1024 * 1024 // (n * n * 8))
    failures, failure_indices, max_ratio, precision_rows = (
        0,
        [],
        0.0,
        case / "batch_getri_matrices.csv.gz",
    )
    with gzip.open(precision_rows, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["matrix", "info", "expected_info", "ratio", "cpu_ratio", "limit", "passed"]
        )
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            local_lu = np.asarray(factors[start:stop])
            local_pivots = np.asarray(pivots[start:stop])
            diagonal = local_lu[:, np.arange(n), np.arange(n)]
            zeros = diagonal == 0
            expected = np.where(zeros.any(axis=1), zeros.argmax(axis=1) + 1, 0).astype(
                np.int32
            )
            if meta["mode"] == 2:
                for index in (1, 2, 3):
                    if start <= index < stop:
                        expected[index - start] = (
                            (-2 if index == 1 else -5)
                            if current_numbering
                            else (-3 if index == 1 else -6)
                        )
            if meta["mode"] == 3 and start == 0:
                expected[0] = -4 if current_numbering else -5
            measured, baseline, limits = (
                np.full(stop - start, np.nan) for _ in range(3)
            )
            passed = np.asarray(infos[start:stop]) == expected
            valid = np.flatnonzero(expected == 0)
            if len(valid):
                selected = np.ascontiguousarray(local_lu[valid])
                selected_pivots = np.ascontiguousarray(local_pivots[valid])
                if not meta["pivot"]:
                    selected_pivots[:] = np.arange(1, n + 1, dtype=np.int32)
                cpu = np.empty_like(selected)
                cpu_info = np.empty(len(valid), np.int32)
                reference.inverse(selected, selected_pivots, cpu, cpu_info)
                lu = selected[:, :, :n].transpose(0, 2, 1).astype(np.float64)
                a = (np.tril(lu, -1) + np.eye(n)) @ np.triu(lu)
                indices = np.arange(len(valid))
                for k in range(n - 1, -1, -1):
                    targets = selected_pivots[:, k] - 1
                    saved = a[:, k].copy()
                    a[:, k] = a[indices, targets]
                    a[indices, targets] = saved
                measured[valid] = ratio_batch(
                    a, np.asarray(outputs[start:stop])[valid, :, :n].transpose(0, 2, 1)
                )
                baseline[valid] = ratio_batch(a, cpu[:, :, :n].transpose(0, 2, 1))
                limits[valid] = np.maximum(5 * baseline[valid], 0.05)
                passed[valid] &= (
                    np.isfinite(measured[valid])
                    & (measured[valid] <= limits[valid])
                    & (cpu_info == 0)
                )
                max_ratio = max(max_ratio, float(np.max(measured[valid])))
            failures += int(np.count_nonzero(~passed))
            failure_indices.extend((np.flatnonzero(~passed) + start).tolist())
            for index in range(stop - start):
                writer.writerow(
                    [
                        start + index,
                        infos[start + index],
                        expected[index],
                        None if np.isnan(measured[index]) else measured[index],
                        None if np.isnan(baseline[index]) else baseline[index],
                        None if np.isnan(limits[index]) else limits[index],
                        bool(passed[index]),
                    ]
                )
    result = dict(
        case=case.name,
        **meta,
        matrices_total=batch,
        matrices_passed=batch - failures,
        failed_indices=failure_indices,
        max_ratio=max_ratio,
        correctness_passed=bool(meta["structural_passed"] and failures == 0),
        input_sha256=digest(case / "input.bin"),
        output_sha256=digest(case / "output.bin"),
        reference_library=str(reference.path.resolve()),
        reference_sha256=digest(reference.path),
        reference_symbols=["scipy_sgetri_"],
        precision_csv=precision_rows.name,
        source_manifest_sha256=digest(case / "source_manifest.json"),
    )
    summaries = list(case.rglob("op_summary_*.csv"))
    if summaries:
        assert len(summaries) == 1
        with summaries[0].open() as stream:
            tasks = [
                row for row in csv.DictReader(stream) if "lu950_" in row["Op Name"]
            ]
        tasks.sort(key=lambda row: float(row["Task Start Time(us)"]))
        calls = meta["warmup"] + meta["samples"]
        assert meta["samples"] == 60 and len(tasks) % calls == 0
        kernels = len(tasks) // calls
        parallel = any("lu950_getri_batch_stage" in row["Op Name"] for row in tasks)
        has_info_fill = any("lu950_fill_info" in row["Op Name"] for row in tasks)
        if n <= 128:
            expected_kernels = 2 if has_info_fill else 1
        else:
            elements = n * ((n + 7) // 8 * 8)
            status_bytes = (
                ((math.ceil(n / 32) * 32 + 511) // 512 * 512) if has_info_fill else 512
            )
            matrix_bytes = (4 * elements * 4 + 511) // 512 * 512 + status_bytes
            count = min(batch, max(1, 512 * 1024 * 1024 // matrix_bytes))
            expected_kernels = (
                (4 * math.ceil(n / 32) + int(has_info_fill)) * math.ceil(batch / count)
                if parallel
                else batch * 4 * math.ceil(n / 32)
            )
        assert kernels == expected_kernels, (kernels, expected_kernels)
        samples = []
        for call in range(meta["warmup"], calls):
            samples.append(
                sum(
                    float(row["Task Duration(us)"])
                    for row in tasks[call * kernels : (call + 1) * kernels]
                )
            )
        assert len(samples) == 60 and np.isfinite(samples).all() and min(samples) > 0
        result.update(
            npu_mean60_us=float(np.mean(samples)),
            npu_trimmed50_us=float(np.mean(sorted(samples)[:50])),
            kernels_per_call=kernels,
            timing_scope="msprof sum of all kernels per complete API call; excludes Host and transfers",
        )
        matching = [
            item
            for item in baselines
            if item["n"] == n
            and item["batch"] == batch
            and item["lda"] == item["ldc"] == ld
            and item.get("perf")
        ]
        if matching:
            gpu = matching[0]["perf"]["avg_ms"] * 1000
            result.update(
                gpu_us=gpu,
                npu_limit_us=gpu / 0.35,
                performance_met=bool(np.mean(samples) <= gpu / 0.35),
            )
        with (case / "batch_getri_samples.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample", "all_kernels_us", "retained_in_trimmed50"])
            retained = set(np.argsort(samples, kind="stable")[:50].tolist())
            writer.writerows(
                (index, value, index in retained) for index, value in enumerate(samples)
            )
    (case / "batch_getri_analysis2.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--reference-bridge", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = Reference(args.reference_bridge)
    baselines = json.loads(args.baseline.read_text())["cases"]
    results = []
    for case in sorted(args.root.glob(args.pattern)):
        if (case / "batch_getri_application.json").exists() and (
            case / "output.bin"
        ).exists():
            results.append(analyze(case, reference, baselines))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
