# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Host tests for parameter extents and nested/concurrent collector event ownership."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import types
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "python/msmemscope"
package = types.ModuleType("triton_test_package")
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(f"{package.__name__}.triton_collection", SOURCE / "triton_collection.py")
collector_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collector_module
spec.loader.exec_module(collector_module)


class Tensor:
    def __init__(self, shape=(4,), strides=(1,), address=4096, storage_size=16):
        self.shape = shape
        self.strides = strides
        self.address = address
        self.storage_size = storage_size

    def stride(self):
        return self.strides

    def data_ptr(self):
        return self.address

    def element_size(self):
        return 4

    def untyped_storage(self):
        return types.SimpleNamespace(data_ptr=lambda: 4096, nbytes=lambda: self.storage_size)


class Metadata:
    def __init__(self, call_id=None):
        self.payload = {} if call_id is None else {"_memscope_triton": {"call_id": call_id, "accesses": []}}

    def get(self):
        return self.payload


class TestParameterExtent(unittest.TestCase):
    def test_contiguous(self):
        self.assertEqual(collector_module.tensor_extent(Tensor()), (4096, 16))

    def test_scalar(self):
        self.assertEqual(collector_module.tensor_extent(Tensor((), (), storage_size=4)), (4096, 4))

    def test_empty_never_dereferences_storage(self):
        tensor = Tensor((0,), address=0, storage_size=0)
        self.assertIsNone(collector_module.tensor_extent(tensor))

    def test_strided_offset(self):
        self.assertEqual(collector_module.tensor_extent(Tensor((3,), (2,), 4100, 24)), (4100, 20))

    def test_broadcast_span(self):
        self.assertEqual(collector_module.tensor_extent(Tensor((128,), (0,), storage_size=4)), (4096, 4))

    def test_invalid_extents(self):
        for tensor in (
            Tensor(strides=(-1,)), Tensor(address=0), Tensor(storage_size=8),
            Tensor(address=4092), Tensor(address=(1 << 64) - 4), Tensor(shape=(1 << 62,)),
        ):
            with self.subTest(tensor=vars(tensor)), self.assertRaises(ValueError):
                collector_module.tensor_extent(tensor)


class TestLaunchOwnership(unittest.TestCase):
    def setUp(self):
        self.collector = collector_module.TritonCollector()
        self.collector.enabled = True
        self.events = []
        self.collector._emit = self.events.append

    def test_missing_nested_metadata_does_not_close_outer(self):
        outer, inner = Metadata(1), Metadata()
        self.collector._enter(outer)
        self.collector._enter(inner)
        self.collector._exit(inner)
        self.assertEqual(len(self.events), 1)
        self.collector._exit(outer)
        self.assertEqual([event["phase"] for event in self.events], ["begin", "end"])

    def test_mismatched_exit_does_not_close_launch(self):
        outer = Metadata(1)
        self.collector._enter(outer)
        self.collector._exit(Metadata(2))
        self.assertEqual(len(self.events), 1)
        self.collector._exit(outer)
        self.assertEqual(len(self.events), 2)

    def test_disabled_emits_nothing(self):
        self.collector.enabled = False
        metadata = Metadata(1)
        self.collector._enter(metadata)
        self.collector._exit(metadata)
        self.assertEqual(self.events, [])

    def test_end_closes_already_started_launch_after_disable(self):
        metadata = Metadata(1)
        self.collector._enter(metadata)
        self.collector.disable()
        self.collector._exit(metadata)
        self.assertEqual([event["phase"] for event in self.events], ["begin", "end"])

    def test_thread_local_pairing(self):
        def launch(index):
            metadata = Metadata(index)
            self.collector._enter(metadata)
            self.collector._exit(metadata)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(launch, range(40)))
        for index in range(40):
            phases = [event["phase"] for event in self.events if event["call_id"] == index]
            self.assertEqual(phases, ["begin", "end"])


if __name__ == "__main__":
    unittest.main()
