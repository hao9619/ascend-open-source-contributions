#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Resolve changed BLAS API declarations; return no selection when scope is uncertain."""

from pathlib import Path
import logging
import re
import subprocess
import sys


LOGGER = logging.getLogger(__name__)
LEXEMES = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r"[A-Za-z_]\w*|\d+|[^\s]"
)
PROTOTYPE = re.compile(r"\baclblasStatus_t\s+(aclblas\w+)\s*\([^;{}]*\)\s*;")
EMPTY_C_LINKAGE = re.compile(
    r'#ifdef[ \t]+__cplusplus\s+extern[ \t]+"C"[ \t]*\{\s*#endif\s*'
    r'#ifdef[ \t]+__cplusplus\s*\}\s*#endif'
)
C_OPEN = re.compile(r'#ifdef[ \t]+__cplusplus\s+extern[ \t]+"C"[ \t]*\{\s*#endif')
C_CLOSE = re.compile(r'#ifdef[ \t]+__cplusplus\s*\}\s*#endif')


def without_comments(text):
    return LEXEMES.sub(lambda match: " " if match[0].startswith(("//", "/*")) else match[0], text)


def tokens(text):
    return tuple(LEXEMES.findall(text))


def declarations(text):
    clean = without_comments(text)
    apis = {}
    for match in PROTOTYPE.finditer(clean):
        if re.search(r"(?m)^\s*#", match[0]):
            raise ValueError("conditional API declaration requires full regression")
        name = match[1]
        prefix = clean[:match.start()]
        depth = len(C_OPEN.findall(prefix)) - len(C_CLOSE.findall(prefix))
        declaration = tokens(match[0]), depth
        if name in apis and apis[name] != declaration:
            raise ValueError("ambiguous declaration: " + name)
        apis[name] = declaration
    return apis, clean


def skeleton(clean, changed):
    # Keep existing API positions, so moving declarations across linkage scopes is detected.
    seen = set()

    def replace(match):
        name = match[1]
        if name in changed or name in seen:
            return ""
        seen.add(name)
        return "@" + name + ";"

    rest = PROTOTYPE.sub(replace, clean)
    # A newly added API may bring its own __cplusplus/extern-C wrapper. Once its
    # declaration is removed, that exact empty wrapper has no remaining effect.
    rest = EMPTY_C_LINKAGE.sub("", rest)
    directives = tuple(line for line in rest.splitlines() if line.lstrip().startswith("#"))
    return directives, tokens(rest)


def changed_ops(root, before, after):
    old, old_clean = declarations(before)
    new, new_clean = declarations(after)
    if old.keys() - new.keys():
        raise ValueError("removed APIs changed")
    changed = [name for name in sorted(new) if old.get(name) != new[name]]
    if not changed:
        raise ValueError("no semantic API declaration change")
    if skeleton(old_clean, changed) != skeleton(new_clean, changed):
        raise ValueError("shared definitions, directives or declaration scopes changed")
    result = []
    for name in changed:
        op = re.sub(r"([A-Z])", r"_\1", name[len("aclblas"):]).lstrip("_").lower()
        candidates = [root / "test" / op / "CMakeLists.txt"]
        candidates.extend(sorted((root / "test").glob("*/" + op + "/CMakeLists.txt")))
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise ValueError("no unambiguous test target for " + name)
        if op not in result:
            result.append(op)
    return result


def log_selection(result):
    result_logger = logging.getLogger(__name__ + ".selection")
    result_logger.setLevel(logging.INFO)
    result_logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    result_logger.addHandler(handler)
    try:
        result_logger.info("%s", ",".join(result))
    finally:
        result_logger.removeHandler(handler)
        handler.close()


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if len(sys.argv) != 4:
        LOGGER.error("usage: changed_header_ops.py <base> <head> <header>")
        return 1
    base, head, header = sys.argv[1:]
    # Other shared headers retain the full-regression fallback.
    if header != "include/cann_ops_blas.h":
        LOGGER.warning("Shared header requires full regression: %s", header)
        return 1
    root = Path(__file__).resolve().parents[2]
    try:
        def read(ref):
            return subprocess.check_output(
                ["git", "show", ref + ":" + header], cwd=root, stderr=subprocess.PIPE
            ).decode("utf-8")
        result = changed_ops(root, read(base), read(head))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        LOGGER.warning("Header selection unavailable: %s", error)
        return 1
    # The dedicated logger keeps machine-readable output separate from diagnostics.
    log_selection(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
