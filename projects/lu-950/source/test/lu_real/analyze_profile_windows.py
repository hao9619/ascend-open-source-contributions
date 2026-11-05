# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Associate msprof device durations with serial, synchronized API call windows."""

import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(results, profiler, single_case=False, stream_ids=False):
    cases = [json.loads(line) for line in results.read_text().splitlines()]
    cases = [case for case in cases if "case_index" in case]
    windows = []
    for case_position, case in enumerate(cases):
        for window in case.get("call_windows", []):
            windows.append(
                dict(**window, case_position=case_position, durations=[], names=[])
            )
    windows.sort(key=lambda window: window["begin_utc_ns"])
    if not windows:
        raise ValueError("No call windows; rerun validation with --record-windows")
    for previous, current in zip(windows, windows[1:]):
        if previous["end_utc_ns"] > current["begin_utc_ns"]:
            raise ValueError(
                "Overlapping API windows cannot be attributed unambiguously"
            )
    starts = [window["begin_utc_ns"] / 1000 for window in windows]
    csv_files = sorted(profiler.rglob("*op_summary*.csv"))
    if not csv_files:
        raise ValueError("No msprof op_summary CSV found")
    tasks = []
    for file in csv_files:
        with file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                name = row.get("Op Name", "")
                if "lu950_" not in name:
                    continue
                tasks.append(row)
    tasks.sort(key=lambda row: float(row["Task Start Time(us)"]))
    kernel_count = len(tasks)
    if stream_ids:
        identifiers = [case["device_stream_id"] for case in cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Every case requires a distinct retained stream")
        if any(int(row["Stream ID"]) not in identifiers for row in tasks):
            raise ValueError("Unattributed LU950 kernels on an unexpected stream")
        for case_position, stream_id in enumerate(identifiers):
            selected = [row for row in tasks if int(row["Stream ID"]) == stream_id]
            calls = [
                window for window in windows if window["case_position"] == case_position
            ]
            if len(calls) != 70 or not selected or len(selected) % 70:
                raise ValueError(f"Incomplete 70-call stream capture for {stream_id}")
            per_call = len(selected) // 70
            names = [row["Op Name"] for row in selected[:per_call]]
            for position, window in enumerate(calls):
                group = selected[position * per_call : (position + 1) * per_call]
                if [row["Op Name"] for row in group] != names:
                    raise ValueError("Kernel sequence varies between repeated calls")
                window["durations"] = [float(row["Task Duration(us)"]) for row in group]
                window["names"] = names
    elif single_case:
        if len(cases) != 1 or len(windows) != 70 or not tasks or len(tasks) % 70:
            raise ValueError(
                "Single-case profiling requires exactly one case and 70 complete calls"
            )
        per_call = len(tasks) // 70
        expected_names = [row["Op Name"] for row in tasks[:per_call]]
        for position, window in enumerate(windows):
            selected = tasks[position * per_call : (position + 1) * per_call]
            if [row["Op Name"] for row in selected] != expected_names:
                raise ValueError("Kernel sequence varies between identical API calls")
            window["durations"] = [float(row["Task Duration(us)"]) for row in selected]
            window["names"] = expected_names
    else:
        for row in tasks:
            name = row["Op Name"]
            start = float(row["Task Start Time(us)"])
            duration = float(row["Task Duration(us)"])
            position = bisect.bisect_right(starts, start) - 1
            if (
                position < 0
                or start + duration > windows[position]["end_utc_ns"] / 1000
            ):
                raise ValueError(
                    f"Kernel outside recorded API windows: {name}, {start}"
                )
            windows[position]["durations"].append(duration)
            windows[position]["names"].append(name)
    output = []
    for case_position, case in enumerate(cases):
        calls = [
            window for window in windows if window["case_position"] == case_position
        ]
        counts = [len(window["durations"]) for window in calls]
        measured = [
            sum(window["durations"]) for window in calls if not window["warmup"]
        ]
        if (
            len(calls) != 70
            or len(measured) != 60
            or not counts
            or min(counts) == 0
            or len(set(counts)) != 1
        ):
            raise ValueError(
                f"Incomplete or inconsistent 10+60 kernel capture for case {case['case_index']}: {counts}"
            )
        mean = sum(measured) / 60
        trimmed = sum(sorted(measured)[:50]) / 50
        perf = case["parameters"].get("perf")
        gpu = perf["avg_ms"] * 1000 if perf else None
        threshold = gpu / 0.35 if gpu is not None else None
        output.append(
            dict(
                kind=case["kind"],
                case_index=case["case_index"],
                parameters=case["parameters"],
                correctness_passed=case["passed"],
                kernels_per_call=counts[0],
                warmup=10,
                samples=60,
                samples_us=measured,
                npu_mean60_us=mean,
                npu_trimmed50_us=trimmed,
                gpu_us=gpu,
                npu_limit_us=threshold,
                performance_met=mean <= threshold if threshold is not None else None,
                usable=bool(
                    case["passed"] and threshold is not None and mean <= threshold
                ),
                kernel_names=sorted(
                    set(name for window in calls for name in window["names"])
                ),
            )
        )
    return dict(
        scope="msprof sum of every LU950 kernel per complete API; Host and transfer durations excluded",
        attribution=(
            "Unique retained stream IDs per case; 70 identical ordered kernel sequences; device durations only"
            if stream_ids
            else "One case per profiler process; 70 identical ordered kernel sequences; device durations only"
            if single_case
            else "Host UTC windows identify calls only; all durations come from msprof CSV"
        ),
        result_sha256=sha256(results),
        profiler_csv=[dict(path=str(file), sha256=sha256(file)) for file in csv_files],
        all_kernel_rows_assigned=True,
        kernel_rows=kernel_count,
        cases=output,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--profiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--single-case", action="store_true")
    mode.add_argument("--stream-ids", action="store_true")
    args = parser.parse_args()
    result = analyze(args.results, args.profiler, args.single_case, args.stream_ids)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                cases=len(result["cases"]),
                kernel_rows=result["kernel_rows"],
                correctness_failed=[
                    row["case_index"]
                    for row in result["cases"]
                    if not row["correctness_passed"]
                ],
                performance_failed=[
                    row["case_index"]
                    for row in result["cases"]
                    if row["performance_met"] is False
                ],
                missing_gpu_baseline=[
                    row["case_index"]
                    for row in result["cases"]
                    if row["performance_met"] is None
                ],
            )
        )
    )
    return int(
        any(
            row["performance_met"] is False or not row["correctness_passed"]
            for row in result["cases"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
