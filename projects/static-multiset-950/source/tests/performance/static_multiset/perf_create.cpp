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
using Context = LifecycleContext<Key>;

template <typename Key>
void SetupCreate(std::uint64_t capacity)
{
    GetBenchmarkContext<Context<Key>>().capacity = static_cast<std::size_t>(capacity);
}

template <typename Key>
TestResult TestCreate()
{
    auto& context = GetBenchmarkContext<Context<Key>>();
    std::optional<aclco::StaticMultiset<Key>> set;
    auto result = Measure([&] { set.emplace(MakeSet<Key>(context.capacity, context.streamGuard.stream)); });
    set.reset();
    return result;
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultisetCreateI32, (TestCreate<std::int32_t>), (SetupCreate<std::int32_t>),
                          std::uint64_t);
REGISTER_PERFORMANCE_TEST(staticMultisetCreateI64, (TestCreate<std::int64_t>), (SetupCreate<std::int64_t>),
                          std::uint64_t);

REGISTER_PERFORMANCE_ARGS(staticMultisetCreateI32, "StaticMultiset create (Capacity) I32",
                          (std::initializer_list<std::tuple<std::uint64_t>>{{CAPACITY}}), std::uint64_t);
REGISTER_PERFORMANCE_ARGS(staticMultisetCreateI64, "StaticMultiset create (Capacity) I64",
                          (std::initializer_list<std::tuple<std::uint64_t>>{{CAPACITY}}), std::uint64_t);
} // namespace aclco::test
