#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ops-blas 用例执行脚本，根据修改文件触发对应算子的测试用例
#
# 用例级增量（issue #457）：
#   - PR 修改某算子的 *_test.csv        → 只运行该 CSV 中被修改的用例（case_name）
#   - PR 修改算子的 .cpp/.h（内核/tiling/测试代码）→ 运行该算子全量用例
#   - 传入 all（或 run_all=true，公共文件变更）→ 全仓全量用例（手动回归用）
#
# 用法：
#   bash run_example.sh <pr_filelist.txt> [soc_version]           # 增量（默认）
#   bash run_example.sh all [soc_version]                          # 全量（手动执行）
ROOT_PATH=$(
    cd "$(dirname $0)"/../..
    pwd
)
cd ${ROOT_PATH}
if [ $# -lt 1 ]; then
    echo "Usage: $0 <pr_filelist.txt|all> [soc_version]"
    exit 1
fi

pr_filelist=$1
soc_version=$2

# all 参数：手动执行时运行全仓全量用例（issue #457）
# RUN_FLAGS 支持的值（本脚本消费，不会透传给 build.sh）：
#   --no-smoke  关闭冒烟抽样，执行真正全量用例
#   --blas-all  使用合并 Runner 单进程执行（suite 初始化 80 次 → 1 次）
if [ "$pr_filelist" = "all" ]; then
    echo "Run all tests (manual full regression)."
    extra_flags=""
    if [[ "${RUN_FLAGS:-}" == *"--no-smoke"* ]]; then
        echo "Smoke sampling disabled by RUN_FLAGS, running full cases."
    else
        extra_flags="${extra_flags} --smoke"
    fi
    if [[ "${RUN_FLAGS:-}" == *"--blas-all"* ]]; then
        extra_flags="${extra_flags} --blas-all"
    fi
    if [ -n "$soc_version" ]; then
        bash build.sh --run --soc=$soc_version ${extra_flags}
    else
        bash build.sh --run ${extra_flags}
    fi
    exit $?
fi

if [ ! -f "$pr_filelist" ]; then
    echo "Error: File $pr_filelist not found"
    exit 1
fi

ops=()
# case 选择映射：op -> 该算子被 PR 修改的用例名集合（CSV 驱动用例级增量）
declare -A op_cases
run_all=false
ignore_dirs="common include utils frame"

resolve_diff_base() {
    local base_branch="origin/${TARGET_BRANCH:-master}"
    local parent_commit=""
    if git cat-file -e "${base_branch}^{commit}" 2>/dev/null; then
        echo "$base_branch"
        return
    fi
    if git cat-file -e HEAD~1 2>/dev/null; then
        echo "HEAD~1"
        return
    fi
    # A depth-one checkout retains parent IDs in the raw commit, but not their objects.
    parent_commit=$(git cat-file -p HEAD 2>/dev/null | sed -n 's/^parent //p' | head -n 1)
    if [[ "$parent_commit" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
        if git cat-file -e "${parent_commit}^{commit}" 2>/dev/null; then
            echo "$parent_commit"
            return
        fi
        echo "Fetching comparison baseline $parent_commit for shallow checkout." >&2
        if git fetch --no-tags --depth=1 origin "$parent_commit" >&2; then
            echo "$parent_commit"
            return
        fi
    fi
    # Keep the conservative full-regression fallback when no baseline can be obtained.
    echo "HEAD~1"
}

# 公共头文件变更（issue #457）：include/cann_ops_blas.h 等被新增算子 PR 触发时，
# 全仓用例可能超出 PreSmoke 的任务时限。库源码仍完整编译，测试目标按变更选择：
# 比较完整 API 声明，忽略注释和声明排版，支持 test/<family>/<op> 的测试目录。
# 共享定义、宏、删除 API 或无法映射的接口仍保守回退全仓。
include_ops=""
while IFS= read -r filepath; do
    if [[ "$filepath" == *.md ]]; then
        continue
    fi
    if [[ "$filepath" == include/* ]]; then
        base_branch=$(resolve_diff_base)
        header_ops=$(python3 scripts/ci/changed_header_ops.py "$base_branch" HEAD "$filepath" || true)
        if [ -z "$header_ops" ]; then
            echo "Warning: shared header scope unresolved, fallback to run all."
            run_all=true
            break
        fi
        include_ops="$include_ops,$header_ops"
        continue
    fi
    if [[ !"$filepath" =~ blas* ]] && [[ !"$filepath" =~ test* ]];then
        continue
    fi
    paths=($(echo "$filepath" | sed 's/\// /g'))
    second_dir=${paths[2]}
    first_dir=${paths[1]}
    root_dir=${paths[0]}

    if [[ "$root_dir" == "test" ]] && ([[ ! -d "$root_dir/$first_dir" ]] || [[ " $ignore_dirs " =~ " $first_dir " ]]); then
        run_all=true
        break
    fi

    if [[ -n "$second_dir" ]] && [[ -d "./test/$first_dir/$second_dir" ]]; then
        op=$second_dir
    elif [[ -n "$first_dir" ]] && [[ -d "./test/$first_dir" ]]; then
        op=$first_dir
    else
        continue
    fi
    ops+=($op)

    # CSV 变更 → 记录被修改的用例名；非 CSV 变更（代码/配置）→ 该算子跑全量
    case_file="./$filepath"
    if [[ "$filepath" == *_test.csv ]] && [ -f "$case_file" ]; then
        # 该算子已被非 CSV 变更标记 ALL（代码变更覆盖 CSV 增量）时不再追加，
        # 避免往 ALL 哨兵后拼出 "ALL,case1" 破坏 BLAS_TEST_CASES 过滤语义
        if [ "${op_cases[$op]:-}" = "ALL" ]; then
            continue
        fi
        # 与 git 基线比对：新增/修改行的 case_name（删除行无法在本分支取到，不纳入）
        # 管道口径：'^[+][^+]' 取新增数据行（排除 +++ 头），awk 剥离 + 前缀后取
        # 首列 case_name，排除 header
        base_branch=$(resolve_diff_base)
        while IFS= read -r case_name; do
            [ -z "$case_name" ] && continue
            [[ "$case_name" == "case_name" ]] && continue
            op_cases[$op]="${op_cases[$op]:-},${case_name}"
        done < <(git diff "$base_branch" HEAD -- "$filepath" 2>/dev/null \
                 | grep -E '^[+][^+]' \
                 | awk -F, '{s=$1; gsub(/^\+/, "", s); print s}' || true)
        # git diff 无基线信息（如 filelist 缺失）时退化为全量
        if [ -z "${op_cases[$op]:-}" ]; then
            echo "Warning: no changed cases parsed from $filepath, will run all cases of $op."
        fi
    else
        # 非 CSV 变更（代码/配置）→ 该算子跑全量。
        # 先清空可能已记录的用例清单：同一算子的 CSV 与非 CSV 文件在
        # filelist 中先后出现时，代码变更意味着整算子行为改变，
        # 必须覆盖（ALL 优先），避免 "ALL,case1" 混合哨兵破坏过滤语义
        op_cases[$op]="ALL"
    fi
done < "$pr_filelist"

# include 公共头变更场景（issue #457）：算子来自 API 函数名推导，用例为全量
if [ -n "$include_ops" ]; then
    include_ops="${include_ops#,}"
    echo "Include-derived ops: $include_ops."
    IFS=',' read -ra _inc_ops <<< "$include_ops"
    for _iop in "${_inc_ops[@]}"; do
        ops+=("$_iop")
        # 头文件只保证 API 面变化，用例无法定位到具体 case，跑该算子全量
        op_cases[$_iop]="ALL"
    done
fi

ops+=("blasLtMatmul")

declare -A _seen
_unique=()
for _op in "${ops[@]}"; do
    if [[ -z "${_seen[$_op]}" ]]; then
        _seen[$_op]=1
        _unique+=("$_op")
    fi
done
ops=("${_unique[@]}")

echo "Trigger Ops: ${ops[@]}."
echo "Need run all: ${run_all}."

# 多 device 分片（issue #457·方案4）：CI runner 标注 npu=N 时按设备分片并行。
# RUNNER_NPU_COUNT 由 workflow 注入（.gitcode/workflows/ops-blas_action.yml），
# 未注入时单设备串行。
extra_flags=""
if [ -n "${RUNNER_NPU_COUNT:-}" ] && [ "${RUNNER_NPU_COUNT}" -gt 1 ] 2>/dev/null; then
    extra_flags="${extra_flags} --devices=${RUNNER_NPU_COUNT}"
fi

if [ "$run_all" = true ]; then
    # 公共目录变更触发 run_all（issue #457）：全仓完整用例可能超出 PreSmoke 时限。
    # 自动启用 --smoke 分级抽样（负例/边界全保）；手动要全量时
    # RUN_FLAGS=--no-smoke bash run_example.sh all。
    if [[ "${RUN_FLAGS:-}" != *"--no-smoke"* ]]; then
        extra_flags="${extra_flags} --smoke"
    fi
    if [ -n "$soc_version" ]; then
        cmd="bash build.sh --run --soc=$soc_version ${extra_flags}"
    else
        cmd="bash build.sh --run ${extra_flags}"
    fi
else
    # 新接口仍使用现有 smoke 抽样；--no-smoke 保留手动完整回归入口。
    if [ -n "$include_ops" ] && [[ "${RUN_FLAGS:-}" != *"--no-smoke"* ]]; then
        extra_flags="${extra_flags} --smoke"
    fi
    ops_str=""
    cases_env=""
    for op in "${ops[@]}"; do
        if [ -z "$ops_str" ]; then
            ops_str="$op"
        else
            ops_str="$ops_str,$op"
        fi
        # 用例级增量：非 ALL 且非空的算子记录其用例清单；ALL/空 跑全量
        # 环境变量格式（与 test_main.cpp 解析约定一致）："op1:case1,case2;op2:case3;"
        op_case_list="${op_cases[$op]:-}"
        if [ -n "$op_case_list" ] && [ "$op_case_list" != "ALL" ]; then
            # 值内首字符为分隔逗号，op 与用例列表之间以冒号连接
            cases_env="${cases_env}${op}:${op_case_list#,};"
        fi
    done
    export BLAS_TEST_CASES="${cases_env}"
    echo "Case filter: ${BLAS_TEST_CASES:-<none, run all cases>}."
    if [ -n "$soc_version" ]; then
        cmd="bash build.sh --run --ops=$ops_str --soc=$soc_version ${extra_flags}"
    else
        cmd="bash build.sh --run --ops=$ops_str ${extra_flags}"
    fi
fi

echo "Command: $cmd."
${cmd}
