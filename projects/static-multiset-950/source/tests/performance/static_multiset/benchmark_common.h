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
#pragma once
#include "../performance_test_framework.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "static_multiset.h"

namespace aclco::test::multiset_benchmark {
constexpr std::uint64_t NUM_INPUTS = 100000000ULL;
constexpr std::uint64_t CAPACITY = 100000000ULL;
constexpr double OCCUPANCY = 0.5;
constexpr std::uint64_t MULTIPLICITY = 1ULL;

template <typename Func>
TestResult Measure(Func&& func)
{
    auto const start = std::chrono::high_resolution_clock::now();
    func();
    auto const stop = std::chrono::high_resolution_clock::now();
    auto const cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    return TestResult(static_cast<double>(cpu), static_cast<double>(cpu), 0);
}

template <typename T>
constexpr T Empty()
{
    return std::numeric_limits<T>::lowest();
}

template <typename Key>
auto MakeSet(std::size_t capacity, aclrtStream stream)
{
    return aclco::StaticMultiset<Key>(aclco::Extent<std::size_t>(capacity), Empty<Key>(), stream);
}

template <typename Key>
std::vector<Key> MakeKeys(std::size_t count)
{
    std::vector<Key> keys(count);
    for (std::size_t i = 0; i < count; ++i)
        keys[i] = static_cast<Key>(i + 1);
    return keys;
}

template <typename Key>
std::vector<Key> MakeQueries(std::size_t count, double matchingRate)
{
    auto const hits = static_cast<std::size_t>(static_cast<double>(count) * matchingRate);
    std::vector<Key> queries(count);
    for (std::size_t i = 0; i < hits; ++i)
        queries[i] = static_cast<Key>(i + 1);
    for (std::size_t i = hits; i < count; ++i)
        queries[i] = static_cast<Key>(count + i + 1);
    return queries;
}

inline std::size_t CheckedCapacity(std::uint64_t numInputs, double occupancy)
{
    if (!std::isfinite(occupancy) || occupancy <= 0.0 || occupancy > 1.0) {
        throw std::invalid_argument("Occupancy must be finite and in (0, 1]");
    }
    const double capacity = static_cast<double>(numInputs) / occupancy;
    // The exclusive power-of-two bound also handles rounding at SIZE_MAX.
    const double upperBound = std::ldexp(1.0, std::numeric_limits<std::size_t>::digits);
    if (!std::isfinite(capacity) || capacity >= upperBound) {
        throw std::overflow_error("Benchmark capacity exceeds size_t");
    }
    return static_cast<std::size_t>(capacity);
}

template <typename Key>
struct InputContext {
    using KeyType = Key;
    AclStreamGuard streamGuard;
    std::size_t numInputs{};
    std::optional<aclco::StaticMultiset<Key>> set;
    DeviceBuffer<Key> keys;
};

template <typename Key>
struct QueryContext : InputContext<Key> {
    DeviceBuffer<Key> queries;
};

template <typename Key, typename Output>
struct OutputContext : QueryContext<Key> {
    DeviceBuffer<Output> output;
};

template <typename Key>
struct RetrieveContext : QueryContext<Key> {
    DeviceBuffer<Key> keyOutput;
    DeviceBuffer<Key> matchOutput;
};

template <typename Key>
struct RetrieveAllContext : InputContext<Key> {
    DeviceBuffer<Key> output;
};

template <typename Key>
struct LifecycleContext {
    AclStreamGuard streamGuard;
    std::size_t capacity{};
};

// Internal linkage preserves the original per-executable/translation-unit context lifetime.
template <typename Context>
static Context& GetBenchmarkContext()
{
    static Context context;
    return context;
}

template <typename Context>
auto PrepareKeys(Context& context, std::uint64_t numInputs, double occupancy, std::uint64_t multiplicity = 1)
{
    if (multiplicity != 1)
        throw std::invalid_argument("Multiplicity must be 1");
    using Key = typename Context::KeyType;
    auto capacity = CheckedCapacity(numInputs, occupancy);
    context.numInputs = static_cast<std::size_t>(numInputs);
    context.set.emplace(MakeSet<Key>(capacity, context.streamGuard.stream));
    auto keys = MakeKeys<Key>(context.numInputs);
    context.keys.Resize(context.numInputs);
    context.keys.CopyFromHostAsync(keys.data(), keys.size(), context.streamGuard.stream);
    // Caller keeps this host allocation alive until the setup stream is synchronized.
    return keys;
}

template <typename Context>
void PopulateSet(Context& context)
{
    context.set->Insert(context.keys.Data(), aclco::Extent<std::size_t>(context.numInputs), context.streamGuard.stream);
}

template <typename Context>
auto PrepareQueries(Context& context, double matchingRate)
{
    auto queries = MakeQueries<typename Context::KeyType>(context.numInputs, matchingRate);
    context.queries.Resize(context.numInputs);
    context.queries.CopyFromHostAsync(queries.data(), queries.size(), context.streamGuard.stream);
    return queries;
}
} // namespace aclco::test::multiset_benchmark
