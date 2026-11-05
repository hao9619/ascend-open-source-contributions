# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run unchanged task cases and attribute complete API device times by stream ID."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
KINDS = ("sgetrf", "sgetri", "sgetribatched", "smatinvbatched")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def execute(command, directory, label, environment):
    with (directory / (label + ".log")).open("x") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
        )
    (directory / (label + "_command.json")).write_text(
        json.dumps(dict(command=command, exit_code=result.returncode), indent=2) + "\n"
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-bridge", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=HERE / "cases")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--indices", help="Comma-separated indices; requires one kind")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=16)
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    if args.indices and len(args.kinds) != 1:
        parser.error("--indices requires exactly one --kinds value")
    if len(set(args.kinds)) != len(args.kinds):
        parser.error("--kinds contains duplicates")
    if any("batched" in kind for kind in args.kinds) and not args.reference_bridge:
        parser.error("Batched verification requires --reference-bridge")
    library, directory = args.library.resolve(), args.output.resolve()
    if not library.is_file():
        parser.error("Library does not exist")
    directory.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    python = [sys.executable, "-I"]
    summary = dict(
        library=str(library),
        library_sha256=digest(library),
        device=args.device,
        warmup=10,
        samples=60,
        timing="msprof sum of every kernel per API; mean60 threshold; trim50 discards largest10",
        chunks=[],
        cases=[],
        stage="running",
    )
    if args.reference_bridge:
        args.reference_bridge = args.reference_bridge.resolve()
        summary["reference_bridge_sha256"] = digest(args.reference_bridge)

    def save():
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    try:
        for kind in args.kinds:
            case_file = (args.cases / ("cases_" + kind + ".json")).resolve()
            count = len(json.loads(case_file.read_text())["cases"])
            indices = (
                [int(item) for item in args.indices.split(",")]
                if args.indices
                else list(range(count))
            )
            if len(set(indices)) != len(indices) or any(
                i < 0 or i >= count for i in indices
            ):
                raise ValueError("Invalid or repeated case index")
            for start in range(0, len(indices), args.chunk):
                selected = indices[start : start + args.chunk]
                label = kind + "_" + str(selected[0])
                results = directory / (label + ".jsonl")
                profiler = directory / (label + "_prof")
                timing = directory / (label + "_timing.json")
                if digest(library) != summary["library_sha256"]:
                    raise RuntimeError("Library changed during the run")
                validator = HERE / (
                    "validate_batch.py" if "batched" in kind else "validate.py"
                )
                command = (
                    python
                    + [
                        str(HERE / "profile_with_summary.py"),
                        "--output=" + str(profiler),
                        "--task-time=on",
                        "--ai-core=off",
                        "--ascendcl=off",
                        "--runtime-api=off",
                        "--sys-lp=off",
                    ]
                    + python
                    + [
                        str(validator),
                        "--library",
                        str(library),
                        "--cases",
                        str(case_file),
                        "--kind",
                        kind,
                        "--output",
                        str(results),
                        "--device",
                        str(args.device),
                        "--indices",
                        ",".join(map(str, selected)),
                        "--warmup",
                        "10",
                        "--samples",
                        "60",
                        "--record-windows",
                        "--stream-per-case",
                    ]
                )
                if "batched" in kind:
                    command += ["--reference-bridge", str(args.reference_bridge)]
                summary["current"] = dict(kind=kind, indices=selected)
                save()
                code = execute(command, directory, label, environment)
                if code:
                    raise RuntimeError(f"Profiler process failed: {label}, code {code}")
                rows = [json.loads(line) for line in results.read_text().splitlines()]
                rows = [row for row in rows if "case_index" in row]
                if [row["case_index"] for row in rows] != selected:
                    raise RuntimeError(f"Incomplete case results: {label}")
                code = execute(
                    python
                    + [
                        str(HERE / "analyze_profile_windows.py"),
                        "--stream-ids",
                        "--results",
                        str(results),
                        "--profiler",
                        str(profiler),
                        "--output",
                        str(timing),
                    ],
                    directory,
                    label + "_analysis",
                    environment,
                )
                if not timing.exists():
                    raise RuntimeError(
                        f"Incomplete msprof capture: {label}, code {code}"
                    )
                measured = json.loads(timing.read_text())["cases"]
                if [row["case_index"] for row in measured] != selected:
                    raise RuntimeError("Timing/result index mismatch")
                failures = [row["case_index"] for row in rows if not row["passed"]]
                slow = [
                    row["case_index"]
                    for row in measured
                    if row["performance_met"] is False
                ]
                unknown = [
                    row["case_index"]
                    for row in measured
                    if row["performance_met"] is None
                ]
                summary["chunks"].append(
                    dict(
                        kind=kind,
                        indices=selected,
                        precision_failures=failures,
                        performance_failures=slow,
                        missing_gpu_baseline=unknown,
                    )
                )
                summary["cases"].extend(measured)
                print(json.dumps(summary["chunks"][-1]), flush=True)
                if failures or slow:
                    summary["stage"] = "failure_isolated"
                    save()
                    return 1
        unknown = [
            dict(kind=row["kind"], case_index=row["case_index"])
            for row in summary["cases"]
            if row["performance_met"] is None
        ]
        summary.update(
            stage="done",
            missing_gpu_baseline=unknown,
            all_performance_known=not unknown,
            cases_executed=len(summary["cases"]),
        )
        save()
        return 3 if unknown else 0
    except Exception:
        summary.update(stage="error", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
