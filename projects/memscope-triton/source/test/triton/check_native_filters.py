# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Check all persisted sessions after the native application has exited and flushed."""

import csv
import json
from pathlib import Path
import re
import sys

folder = Path(sys.argv[1])
expected = json.loads((folder / "expected.json").read_text())
rows = []
for path in folder.rglob("memscope_dump_*.csv"):
    with path.open(newline="") as stream:
        rows.extend(csv.DictReader(stream))
mode = expected["mode"]
kinds = ({"TRITON_START", "TRITON_END"} if mode in ("launch", "both") else set())
kinds |= {"READ", "WRITE"} if mode in ("access", "both") else set()
collected = [row for row in rows if row["Name"] == "filtered_copy"]
assert len(collected) == 3 * len(kinds), (mode, len(collected), kinds)
for before, after in expected["windows"]:
    group = [row for row in collected if before <= int(row["Timestamp(ns)"]) <= after]
    assert {row["Event Type"] for row in group} == kinds, (mode, group)
    assert len(group) == len(kinds)
    if not kinds:
        continue
    ids = {re.search(r"call_id:(\d+)", row["Attr"]).group(1) for row in group}
    assert len(ids) == 1
    events = {row["Event Type"]: row for row in group}
    for kind, parameter in (("READ", "x"), ("WRITE", "output")):
        if kind in events:
            assert int(events[kind]["Ptr"], 16) == expected[parameter]
            assert int(re.search(r"(?:\{|,)size:(\d+)", events[kind]["Attr"]).group(1)) == expected["size"]
    if mode == "both":
        assert int(events["TRITON_START"]["Timestamp(ns)"]) <= int(events["READ"]["Timestamp(ns)"])
        assert int(events["WRITE"]["Timestamp(ns)"]) <= int(events["TRITON_END"]["Timestamp(ns)"])
aten = [row for row in rows if row["Name"] == "torch._ops.aten.add.Tensor"]
aten_kinds = {kind.replace("TRITON", "ATEN") for kind in kinds}
assert {row["Event Type"] for row in aten} == aten_kinds, (mode, aten)
print(f"NATIVE FILTER CSV PASS: {mode}; 3 sessions; timestamps inside caller windows; disabled calls absent")
