# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Repeat full-spec I2V generation and keep each synchronized Host timing."""

import json
import os
import time


def run_full_spec_samples(run_once, synchronize, full_warmups, measured_runs, jsonl_path,
                          before_run=None, measure_memory=None):
    """Return the last video and individual warmup/sample timing records.

    The caller owns model setup and the initial short compilation warmup. A
    fresh `run_once` call is made for every full-spec request, allowing the
    pipeline to reset its random generator and request-scoped cache state.
    """
    if full_warmups < 0 or measured_runs < 1:
        raise ValueError("full_warmups must be non-negative and measured_runs must be positive")
    if not jsonl_path:
        raise ValueError("jsonl_path is required for repeated full-spec samples")

    # Exclusive creation prevents a later run from silently mixing samples.
    output = open(jsonl_path, "x", encoding="utf-8")
    records = []
    video = None
    try:
        for phase, count in (("warmup", full_warmups), ("sample", measured_runs)):
            for run_id in range(1, count + 1):
                if video is not None:
                    del video
                    video = None
                synchronize()
                if before_run is not None:
                    before_run()
                start = time.perf_counter()
                video = run_once()
                synchronize()
                elapsed = time.perf_counter() - start
                record = {"phase": phase, "run_id": run_id, "host_generate_s": elapsed}
                if measure_memory is not None:
                    record.update(measure_memory(video))
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                output.flush()
                os.fsync(output.fileno())
                records.append(record)
    finally:
        output.close()
    return video, records
