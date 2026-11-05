# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Host dataflow safety checks: unfamiliar pointer operations must never imply read-only certainty."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "python/msmemscope/triton_access.py"
spec = importlib.util.spec_from_file_location("effect_safety", SOURCE)
access = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = access
spec.loader.exec_module(access)


class Value:
    def __init__(self, key, kind="!tt.ptr<f32>"):
        self.key, self.kind = key, kind

    def id(self):
        return self.key

    def get_type(self):
        return self.kind


class Operation:
    def __init__(self, name, operands=(), results=()):
        self.name, self.operands, self.results = name, operands, results

    def get_name(self):
        return self.name

    def get_num_operands(self):
        return len(self.operands)

    def get_operand(self, index):
        return self.operands[index]

    def get_num_results(self):
        return len(self.results)

    def get_result(self, index):
        return self.results[index]

    def get_num_regions(self):
        return 0

    def get_block(self):
        return None


def analyze(arguments, operations):
    function = types.SimpleNamespace(
        args=arguments.__getitem__, type=types.SimpleNamespace(param_types=lambda: arguments),
    )
    module = types.SimpleNamespace(
        has_function=lambda name: name == "main", get_function=lambda name: function,
        walk=lambda visitor: [visitor(operation) for operation in operations],
    )
    return [(item.read, item.write, item.unknown) for item in access.analyze_module(module, "main")]


class TestEffectSafety(unittest.TestCase):
    def test_select_unions_pointer_origins_without_tainting_integer_predicate(self):
        x, y, output = Value(1), Value(2), Value(3)
        predicate, selected = Value(4, "i1"), Value(5)
        result = analyze([x, y, output, predicate], [
            Operation("arith.select", (predicate, x, y), (selected,)),
            Operation("tt.load", (selected,)), Operation("tt.store", (output,)),
        ])
        self.assertEqual(result, [(True, False, False), (True, False, False),
                                  (False, True, False), (False, False, False)])

    def test_unrecognized_pointer_consumer_marks_affected_parameter_unknown(self):
        x, output = Value(1), Value(2)
        result = analyze([x, output], [Operation("tt.load", (x,)), Operation("vendor.write", (x,)),
                                      Operation("tt.store", (output,))])
        self.assertEqual(result, [(True, False, True), (False, True, False)])

    def test_unknown_pointer_producer_never_silently_becomes_read_only(self):
        x, output, indirect = Value(1), Value(2), Value(3)
        result = analyze([x, output], [Operation("vendor.pointer", (x,), (indirect,)),
                                      Operation("tt.store", (indirect,)), Operation("tt.load", (output,))])
        self.assertEqual(result, [(False, False, True), (True, False, True)])

    def test_cyclic_alias_graph_terminates_and_preserves_atomic_read_write(self):
        x, a, b = Value(1), Value(2), Value(3)
        result = analyze([x], [Operation("arith.select", (x, b), (a,)), Operation("tt.addptr", (a,), (b,)),
                               Operation("tt.atomic_cas", (b,))])
        self.assertEqual(result, [(True, True, False)])


if __name__ == "__main__":
    unittest.main()
