# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public API parameter errors, empty problems and multi-stream workspace lifetime."""

import argparse
import ctypes as ct
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import Solver  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-probe", type=Path)
    parser.add_argument("--multi-repeat", type=int, default=1)
    args = parser.parse_args()
    solver = Solver(args.library, 0)
    rows = []
    zero = ct.c_void_p()
    try:
        identity = np.eye(2, dtype=np.float32, order="F")
        a = solver.upload(identity)
        c = solver.upload(identity)
        pivots = solver.upload(np.asarray([1, 2], np.int32))
        infos = solver.upload(np.full(4, -777, np.int32))
        ap = solver.upload(np.full(4, a.value, np.uint64))
        cp = solver.upload(np.full(4, c.value, np.uint64))
        work_size = ct.c_int()
        solver.call(
            "aclsolverSgetri_bufferSize",
            solver.handle,
            ct.c_int(2),
            a,
            ct.c_int(2),
            ct.byref(work_size),
        )
        workspace = solver.allocate(work_size.value * 4)

        def check(
            name, function, arguments, expected_status, expected_info=None, info_count=1
        ):
            solver.write(infos, np.full(4, -777, np.int32))
            actual_status = getattr(solver.library, function)(*arguments)
            solver.sync()
            observed = solver.download(infos, np.empty(4, np.int32))[
                :info_count
            ].tolist()
            passed = actual_status == expected_status and (
                expected_info is None or observed == [expected_info] * info_count
            )
            row = dict(
                test=name,
                status=actual_status,
                expected_status=expected_status,
                info=observed,
                expected_info=expected_info,
                passed=passed,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)

        h = solver.handle
        getrf = [h, ct.c_int(2), ct.c_int(2), a, ct.c_int(2), zero, pivots, infos]
        for name, position, replacement, expected in [
            ("getrf_negative_m", 1, ct.c_int(-1), -1),
            ("getrf_negative_n", 2, ct.c_int(-1), -2),
            ("getrf_null_a", 3, zero, -3),
            ("getrf_host_a", 3, ct.c_void_p(identity.ctypes.data), -3),
            ("getrf_invalid_lda", 4, ct.c_int(1), -4),
            ("getrf_negative_lda", 4, ct.c_int(-1), -4),
            ("getrf_host_pivots", 6, ct.c_void_p(identity.ctypes.data), -6),
        ]:
            arguments = getrf.copy()
            arguments[position] = replacement
            check(name, "aclsolverSgetrf", arguments, 3, expected)
        check(
            "getrf_empty",
            "aclsolverSgetrf",
            [h, ct.c_int(0), ct.c_int(2), zero, ct.c_int(1), zero, zero, infos],
            0,
            0,
        )
        getri = [h, ct.c_int(2), a, ct.c_int(2), pivots, workspace, work_size, infos]
        for name, position, replacement, expected in [
            ("getri_negative_n", 1, ct.c_int(-1), -1),
            ("getri_null_a", 2, zero, -2),
            ("getri_invalid_lda", 3, ct.c_int(1), -3),
            ("getri_negative_lda", 3, ct.c_int(-1), -3),
            ("getri_host_pivots", 4, ct.c_void_p(identity.ctypes.data), -4),
            ("getri_null_workspace", 5, zero, -5),
            ("getri_short_lwork", 6, ct.c_int(work_size.value - 1), -6),
            ("getri_negative_lwork", 6, ct.c_int(-1), -6),
        ]:
            arguments = getri.copy()
            arguments[position] = replacement
            check(name, "aclsolverSgetri", arguments, 3, expected)
        check(
            "getri_empty",
            "aclsolverSgetri",
            [h, ct.c_int(0), zero, ct.c_int(1), zero, zero, ct.c_int(0), infos],
            0,
            0,
        )
        batch = [
            h,
            ct.c_int(2),
            ap,
            ct.c_int(2),
            zero,
            cp,
            ct.c_int(2),
            infos,
            ct.c_int(4),
        ]
        for name, position, replacement, expected in [
            ("batch_negative_n", 1, ct.c_int(-1), -1),
            ("batch_host_array", 2, ct.c_void_p(identity.ctypes.data), -2),
            ("batch_invalid_lda", 3, ct.c_int(1), -3),
            ("batch_host_pivots", 4, ct.c_void_p(identity.ctypes.data), -4),
            ("batch_null_carray", 5, zero, -5),
            ("batch_invalid_ldc", 6, ct.c_int(1), -6),
        ]:
            arguments = batch.copy()
            arguments[position] = replacement
            check(name, "aclsolverSgetriBatched", arguments, 3, expected, 4)
        check(
            "batch_empty",
            "aclsolverSgetriBatched",
            [
                h,
                ct.c_int(2),
                zero,
                ct.c_int(2),
                zero,
                zero,
                ct.c_int(2),
                zero,
                ct.c_int(0),
            ],
            0,
        )
        check(
            "batch_n_zero",
            "aclsolverSgetriBatched",
            [
                h,
                ct.c_int(0),
                zero,
                ct.c_int(1),
                zero,
                zero,
                ct.c_int(1),
                infos,
                ct.c_int(4),
            ],
            0,
            0,
            4,
        )
        check(
            "matinv_reject_33",
            "aclsolverSmatinvBatched",
            [h, ct.c_int(33), ap, ct.c_int(33), cp, ct.c_int(33), infos, ct.c_int(4)],
            3,
            -1,
            4,
        )
        check(
            "matinv_empty",
            "aclsolverSmatinvBatched",
            [h, ct.c_int(2), zero, ct.c_int(2), zero, ct.c_int(2), zero, ct.c_int(0)],
            0,
        )
        check("null_handle", "aclsolverSgetrf", [zero, *getrf[1:]], 9)
        for function, arguments in [
            (
                "aclsolverSgetrf_bufferSize",
                [h, ct.c_int(-1), ct.c_int(2), a, ct.c_int(2), ct.byref(work_size)],
            ),
            (
                "aclsolverSgetri_bufferSize",
                [h, ct.c_int(-1), a, ct.c_int(2), ct.byref(work_size)],
            ),
            (
                "aclsolverSgetrf_bufferSize",
                [h, ct.c_int(2), ct.c_int(2), a, ct.c_int(2), zero],
            ),
            ("aclsolverSgetri_bufferSize", [h, ct.c_int(2), a, ct.c_int(2), zero]),
        ]:
            check("query_invalid_" + str(len(rows)), function, arguments, 3)
        matinv = [h, ct.c_int(2), ap, ct.c_int(2), cp, ct.c_int(2), infos, ct.c_int(4)]
        for name, position, replacement, expected in [
            ("matinv_negative_n", 1, ct.c_int(-1), -1),
            ("matinv_host_array", 2, ct.c_void_p(identity.ctypes.data), -2),
            ("matinv_invalid_lda", 3, ct.c_int(1), -3),
            ("matinv_null_carray", 4, zero, -4),
            ("matinv_invalid_ldc", 5, ct.c_int(1), -5),
        ]:
            arguments = matinv.copy()
            arguments[position] = replacement
            check(name, "aclsolverSmatinvBatched", arguments, 3, expected, 4)
        for name, function, arguments, position in [
            ("batch", "aclsolverSgetriBatched", batch, 7),
            ("matinv", "aclsolverSmatinvBatched", matinv, 6),
        ]:
            arguments = arguments.copy()
            arguments[position] = zero
            check(name + "_null_info", function, arguments, 3)
            original_pointers = solver.download(ap, np.empty(4, np.uint64))
            arguments[position] = ap
            check(name + "_info_overlaps_pointer_array", function, arguments, 3)
            preserved = np.array_equal(
                original_pointers, solver.download(ap, np.empty(4, np.uint64))
            )
            rows[-1]["pointer_array_preserved"] = bool(preserved)
            rows[-1]["passed"] &= bool(preserved)
        for trial in range(args.multi_repeat):
            # Large batch calls on two streams must retain separate live scratch.
            n = 129
            large = np.eye(n, dtype=np.float32, order="F") * 2
            la, lc1, lc2 = (
                solver.upload(large),
                solver.upload(large),
                solver.upload(large),
            )
            lap = solver.upload(np.asarray([la.value], np.uint64))
            cp1, cp2 = (
                solver.upload(np.asarray([lc1.value], np.uint64)),
                solver.upload(np.asarray([lc2.value], np.uint64)),
            )
            second, temporary = ct.c_void_p(), ct.c_void_p()
            solver.call("aclrtCreateStream", ct.byref(second), runtime=True)
            solver.call("aclsolverCreate", ct.byref(temporary))
            second_info = solver.allocate(4)
            cache_records = []
            workspace_probe = (
                ct.CDLL(str(args.workspace_probe)) if args.workspace_probe else None
            )
            for stream, output, status_output in [
                (solver.stream, cp1, infos),
                (second, cp2, second_info),
            ]:
                solver.call("aclsolverSetStream", temporary, stream)
                solver.call(
                    "aclsolverSgetriBatched",
                    temporary,
                    ct.c_int(n),
                    lap,
                    ct.c_int(n),
                    zero,
                    output,
                    ct.c_int(n),
                    status_output,
                    ct.c_int(1),
                )
                if workspace_probe:
                    records = np.zeros((4, 5), np.uint64)
                    count = workspace_probe.lu950_test_workspace_state(
                        temporary, ct.c_void_p(records.ctypes.data), ct.c_int(4)
                    )
                    cache_records.append(records[:count].tolist())
            solver.call("aclsolverDestroy", temporary)
            output1 = solver.download(lc1, np.empty_like(large))
            output2 = solver.download(lc2, np.empty_like(large))
            solver.sync()
            solver.call("aclrtSynchronizeStream", second, runtime=True)
            after_sync1 = solver.download(lc1, np.empty_like(large))
            after_sync2 = solver.download(lc2, np.empty_like(large))
            first_info = solver.download(infos, np.empty(4, np.int32)).tolist()
            other_info = solver.download(second_info, np.empty(1, np.int32)).tolist()
            solver.call("aclrtDestroyStream", second, runtime=True)
            expected = np.eye(n, dtype=np.float32) / 2
            passed = bool(
                np.array_equal(output1, expected) and np.array_equal(output2, output1)
            )
            details = dict(
                test="two_stream_workspace_destroy_inflight_" + str(trial),
                passed=passed,
                cache_records=cache_records,
                first_info=first_info,
                second_info=other_info,
                first_max_error=float(np.max(np.abs(output1 - expected))),
                second_max_error=float(np.max(np.abs(output2 - expected))),
                first_mismatches=int(np.count_nonzero(output1 != expected)),
                second_mismatches=int(np.count_nonzero(output2 != expected)),
                after_stream_sync_passed=bool(
                    np.array_equal(after_sync1, expected)
                    and np.array_equal(after_sync2, expected)
                ),
                diagonal1=output1.diagonal()[:8].tolist(),
                diagonal2=output2.diagonal()[:8].tolist(),
            )
            rows.append(details)
            print(json.dumps(details), flush=True)
            np.savez(
                args.output.parent
                / (args.output.stem + "_trial_" + str(trial) + ".npz"),
                initial=large,
                output1=output1,
                output2=output2,
                after_sync1=after_sync1,
                after_sync2=after_sync2,
            )
    finally:
        solver.close()
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                tests=len(rows),
                passed=sum(r["passed"] for r in rows),
                failed=[r["test"] for r in rows if not r["passed"]],
            )
        ),
        flush=True,
    )
    return int(not all(r["passed"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
