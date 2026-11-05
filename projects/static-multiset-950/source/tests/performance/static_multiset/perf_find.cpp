/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "benchmark_common.h"

namespace aclco::test {
namespace {
using namespace multiset_benchmark;

template <typename Key>
using Context = OutputContext<Key, Key>;

template <typename Key>
void SetupFind(std::uint64_t numInputs, double occupancy, double matchingRate)
{
    auto& context = GetBenchmarkContext<Context<Key>>();
    auto keys = PrepareKeys(context, numInputs, occupancy, 1);
    PopulateSet(context);
    auto queries = PrepareQueries(context, matchingRate);
    context.output.Resize(context.numInputs);
    Sync(context.streamGuard.stream);
}

template <typename Key>
TestResult TestFind()
{
    auto& context = GetBenchmarkContext<Context<Key>>();
    return Measure([&] {
        context.set->Find(context.queries.Data(), context.output.Data(), aclco::Extent<std::size_t>(context.numInputs),
                          context.streamGuard.stream);
    });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultisetFindI32, (TestFind<std::int32_t>), (SetupFind<std::int32_t>), std::uint64_t,
                          double, double);
REGISTER_PERFORMANCE_TEST(staticMultisetFindI64, (TestFind<std::int64_t>), (SetupFind<std::int64_t>), std::uint64_t,
                          double, double);

REGISTER_PERFORMANCE_ARGS(
    staticMultisetFindI32, "StaticMultiset find (NumInputs, Occupancy, MatchingRate) I32",
    (std::initializer_list<std::tuple<std::uint64_t, double, double>>{
        {NUM_INPUTS, OCCUPANCY, 0.1}, {NUM_INPUTS, OCCUPANCY, 0.5}, {NUM_INPUTS, OCCUPANCY, 1.0}}),
    std::uint64_t, double, double);
REGISTER_PERFORMANCE_ARGS(
    staticMultisetFindI64, "StaticMultiset find (NumInputs, Occupancy, MatchingRate) I64",
    (std::initializer_list<std::tuple<std::uint64_t, double, double>>{
        {NUM_INPUTS, OCCUPANCY, 0.1}, {NUM_INPUTS, OCCUPANCY, 0.5}, {NUM_INPUTS, OCCUPANCY, 1.0}}),
    std::uint64_t, double, double);
} // namespace aclco::test
