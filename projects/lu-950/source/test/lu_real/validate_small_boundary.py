# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check exact-size existing-LU inverse dispatch across packed row groups and layouts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import Solver
from validate_batch import Reference, run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    solver, reference = Solver(args.library, 0), Reference(args.reference_bridge)
    records = []
    try:
        with (args.output / "results.jsonl").open("x") as stream:
            for n in range(17, 33):
                for batch in (128, 512, 4096):
                    for padding in (0, 8):
                        case = dict(
                            n=n,
                            batch=batch,
                            lda=n + padding,
                            ldc=n + 2 * padding,
                            iters=2,
                            check=True,
                            perf=None,
                        )
                        index = 100000 + len(records)
                        row = run_case(
                            solver,
                            reference,
                            "sgetribatched",
                            case,
                            index,
                            args.output,
                            0,
                            2,
                        )
                        solver.release()
                        records.append(row)
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        print(json.dumps(row), flush=True)
                        if not row["passed"]:
                            raise RuntimeError(
                                "Existing-LU small inverse boundary failed"
                            )
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
    assert summary["tests"] == summary["passed"] == 96
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
