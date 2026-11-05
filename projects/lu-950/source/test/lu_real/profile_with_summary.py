# Copyright (c) 2026 xlvrNlfxKe All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collect full msprof data, then export complete CSV without a large JSON timeline."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    arguments = sys.argv[1:]
    output = [
        value.split("=", 1)[1] for value in arguments if value.startswith("--output=")
    ]
    if len(output) != 1 or any(value.startswith("--type=") for value in arguments):
        raise ValueError("Pass exactly one --output= and leave --type to this wrapper")
    profiler = shutil.which("msprof")
    if not profiler:
        raise RuntimeError("msprof is not available in PATH")
    result = subprocess.run([profiler, "--type=db", *arguments], check=False)
    if result.returncode:
        return result.returncode
    package = Path(os.environ["ASCEND_HOME_PATH"])
    exporter = package / "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
    if not exporter.is_file():
        raise RuntimeError(
            "The installed CANN package does not provide the documented summary exporter"
        )
    result = subprocess.run(
        [
            "python3",
            str(exporter),
            "export",
            "summary",
            "-dir",
            output[0],
            "--format",
            "csv",
        ],
        check=False,
    )
    if result.returncode:
        return result.returncode
    files = list(Path(output[0]).rglob("op_summary*.csv"))
    if not files or any(path.stat().st_size == 0 for path in files):
        raise RuntimeError("No complete kernel summary was exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
