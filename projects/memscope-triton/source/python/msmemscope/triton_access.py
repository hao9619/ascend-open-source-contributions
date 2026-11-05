# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
"""Infer possible parameter memory effects from compiled Triton IR, without rewriting it.

The result describes parameter-level effects, not the dynamic set of addresses selected by masks.
Unknown operations are diagnosed rather than treated as read-only.
"""

from collections import defaultdict
from dataclasses import dataclass
import re
import tempfile


@dataclass(frozen=True)
class ParameterAccess:
    """An IR argument's possible effects and whether those effects are fully understood."""

    read: bool = False
    write: bool = False
    unknown: bool = False


def _operands(operation):
    return [operation.get_operand(index) for index in range(operation.get_num_operands())]


def _results(operation):
    return [operation.get_result(index) for index in range(operation.get_num_results())]


def _pointer(value):
    return "!tt.ptr<" in str(value.get_type())


def _parse_ir(text, target):
    from triton._C.libtriton import ir
    from triton.compiler.compiler import make_backend

    context = ir.context()
    ir.load_dialects(context)
    make_backend(target).load_dialects(context)
    # Ascend 3.2.2 writes this backend-only target attribute into the TTIR snapshot.
    # Its frontend parser does not register hacc; remove only this exact module annotation.
    text = re.sub(r'module attributes \{hacc.target = #hacc.target<"[A-Za-z0-9]+">\}', "module", text)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", encoding="utf-8") as temporary:
        temporary.write(text)
        temporary.flush()
        module = ir.parse_mlir_module(temporary.name, context)
    module.context = context
    return context, module


def _control_edges(operations, edges):
    """Connect loop-carried values and yields using region identity, not textual SSA names."""
    regions = {}
    blocks = {}
    for operation in operations:
        for index in range(operation.get_num_regions()):
            regions[operation.get_region(index).id()] = (operation, index)
        block = operation.get_block()
        if block is not None:
            blocks[block.id()] = block
    region_blocks = {block.get_parent().id(): block for block in blocks.values()}
    for block in blocks.values():
        owner, region_index = regions.get(block.get_parent().id(), (None, None))
        if owner is None:
            continue
        name = owner.get_name()
        values = _operands(owner)
        if name == "scf.for":
            for index in range(1, block.get_num_arguments()):
                edges[block.get_argument(index).id()].add(values[index + 2].id())
        elif name == "scf.while" and region_index == 0:
            for index in range(block.get_num_arguments()):
                edges[block.get_argument(index).id()].add(values[index].id())
    for operation in operations:
        name = operation.get_name()
        if name not in ("scf.yield", "scf.condition"):
            continue
        block = operation.get_block()
        owner, region_index = regions.get(block.get_parent().id(), (None, None))
        if owner is None:
            continue
        values = _operands(operation)
        if name == "scf.condition":
            values = values[1:]
        for result, value in zip(_results(owner), values):
            edges[result.id()].add(value.id())
        if owner.get_name() == "scf.for":
            for index, value in enumerate(values):
                edges[block.get_argument(index + 1).id()].add(value.id())
        elif owner.get_name() == "scf.while":
            destination = region_blocks.get(owner.get_region(1 - region_index).id())
            if destination is not None:
                for index, value in enumerate(values):
                    edges[destination.get_argument(index).id()].add(value.id())


def analyze_module(module, function_name):
    """Analyze an already parsed module; unsupported pointer effects remain explicitly unknown."""
    if not module.has_function(function_name):
        raise ValueError("Compiled entry function is absent from the TTIR snapshot")
    function = module.get_function(function_name)
    arguments = [function.args(index) for index in range(len(function.type.param_types()))]
    origins = {argument.id(): {index} for index, argument in enumerate(arguments) if _pointer(argument)}
    pointer_indices = set().union(*origins.values()) if origins else set()
    operations = []
    module.walk(operations.append)
    edges = defaultdict(set)
    accesses = []
    uncertain_values = []
    unknown_all = False
    transparent = {
        "tt.addptr",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.trans",
        "tt.reshape",
        "tt.join",
        "tt.split",
        "tt.cat",
        "tt.advance",
        "tt.make_tensor_ptr",
        "arith.select",
        "tt.bitcast",
    }
    structural = {"tt.func", "builtin.module", "scf.for", "scf.if", "scf.while", "scf.yield", "scf.condition"}
    memory = {
        "tt.load": (True, False),
        "tt.store": (False, True),
        "tt.atomic_rmw": (True, True),
        "tt.atomic_cas": (True, True),
    }
    for operation in operations:
        name = operation.get_name()
        values = _operands(operation)
        results = _results(operation)
        if name in memory:
            accesses.append((values[0].id(), *memory[name]))
            if any(_pointer(result) for result in results):
                unknown_all = True  # Pointer loaded from device memory has no known parameter extent.
        elif name in transparent:
            # Pointer offsets and select predicates are not pointer aliases.
            sources = {value.id() for value in values if _pointer(value)}
            for result in results:
                if _pointer(result):
                    edges[result.id()].update(sources)
        elif name not in structural:
            uncertain_values.extend(value.id() for value in values if _pointer(value))
            if any(_pointer(result) for result in results):
                unknown_all = True
    _control_edges(operations, edges)
    changed = True
    while changed:
        changed = False
        for destination, sources in edges.items():
            combined = set().union(*(origins.get(source, set()) for source in sources))
            previous = origins.setdefault(destination, set())
            if not combined.issubset(previous):
                previous.update(combined)
                changed = True
    reads, writes, unknown = set(), set(), set()
    for value, read, write in accesses:
        source = origins.get(value, set())
        if not source:
            unknown_all = True
        if read:
            reads.update(source)
        if write:
            writes.update(source)
    for value in uncertain_values:
        source = origins.get(value, set())
        if not source:
            unknown_all = True
        unknown.update(source)
    if unknown_all:
        unknown.update(pointer_indices)
    return tuple(ParameterAccess(index in reads, index in writes, index in unknown) for index in range(len(arguments)))


def analyze_kernel(kernel):
    """Read the compiled snapshot for this specialization, including cache-loaded kernels."""
    text = kernel.asm.get("ttir")
    if not isinstance(text, str):
        raise ValueError("Compiled kernel has no textual TTIR snapshot")
    context, module = _parse_ir(text, kernel.metadata.target)
    # Keep the context alive until all borrowed MLIR handles are no longer in use.
    result = analyze_module(module, kernel.name)
    del module, context
    return result
