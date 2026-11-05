#!/usr/bin/env python3
"""Compare original task wall-clock means against all 44 supplied baselines."""

import argparse
import json
import re
from pathlib import Path

PATTERN = r'Test Case: .*?\((.*?)\) (I32|I64)\s+Parameters: \[(.*?)\].*?Mean CPU Time: ([\d.]+) us'

parser = argparse.ArgumentParser()
parser.add_argument('logs', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
baseline = json.loads((root / 'tests/performance/static_multiset/baselines.json').read_text())
actual = {}
for path in args.logs.glob('static_multiset_perf_*.log'):
    op = path.stem.removeprefix('static_multiset_perf_')
    text = path.read_text()
    if 'All Tests Completed' not in text:
        raise SystemExit(f'Incomplete benchmark: {path}')
    for match in re.finditer(PATTERN, text, re.S):
        names, dtype, values, elapsed = match.groups()
        params = dict(zip([x.strip() for x in names.split(',')], [float(x.strip()) for x in values.split(',')]))
        key = (op, dtype, params.get('MatchingRate'))
        if key in actual:
            raise SystemExit(f'Duplicate result: {key}')
        actual[key] = float(elapsed) / 1000
lines = [
    '# StaticMultiset 性能对标',
    '',
    '采用任务提供的 CPU wall-clock 计时口径；日志中的 Device Time 为同一数值，不能当作独立设备 profiler 时延。',
    '',
    '| 接口 | dtype | 匹配率 | 标杆 ms | 实测 ms | 标杆/实测 | >=0.4 |',
    '| --- | --- | --- | --- | --- | --- | --- |',
]
passed = 0
results = []
for row in baseline['rows']:
    key = (row['op'], row['dtype'], row['matching_rate'])
    if key not in actual:
        raise SystemExit(f'Missing result: {key}')
    elapsed = actual[key]
    ratio = row['baseline_ms'] / elapsed
    ok = ratio >= 0.4
    passed += ok
    results.append(dict(row, actual_ms=elapsed, ratio=ratio, passed=ok))
    lines.append(
        f"| {key[0]} | {key[1]} | {key[2] if key[2] is not None else '-'} | "
        f"{row['baseline_ms']:.6f} | {elapsed:.6f} | {ratio:.3f} | {'PASS' if ok else 'FAIL'} |"
    )
if len(actual) != len(results):
    raise SystemExit('Unexpected extra cases in performance logs')
lines += ['', f'共 {len(results)} 项，达标 {passed} 项，未达标 {len(results) - passed} 项。', '']
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text('\n'.join(lines))
args.output.with_suffix('.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(f'{passed}/{len(results)} performance cases meet threshold; report: {args.output}')
