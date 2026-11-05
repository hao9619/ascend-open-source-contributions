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

/* Test utilities for static_multimap/static_multiset community tasks. */
#pragma once

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "extent.h"
#include "pair.h"
#include "static_multiset.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace aclco::test::multi_cases {

template <typename Key>
constexpr Key EmptyKey()
{
    return std::numeric_limits<Key>::lowest();
}

template <typename Value>
constexpr Value EmptyValue()
{
    return std::numeric_limits<Value>::lowest();
}

/*
 * Keep constructor binding in one place. If the final public constructor adds
 * probing/storage arguments, only these two functions need adjustment.
 */

template <typename Key>
auto MakeMultiset(std::size_t capacity, aclrtStream stream)
{
    return aclco::StaticMultiset<Key>(aclco::Extent<std::size_t>(capacity), EmptyKey<Key>(), stream);
}

template <typename Key>
std::vector<Key> MakeRepeatedKeys(std::size_t distinct, std::size_t multiplicity)
{
    std::vector<Key> keys;
    keys.reserve(distinct * multiplicity);
    for (std::size_t i = 0; i < distinct; ++i) {
        for (std::size_t repeat = 0; repeat < multiplicity; ++repeat) {
            keys.push_back(static_cast<Key>(i + 1));
        }
    }
    return keys;
}

template <typename Key, typename Value>
std::vector<aclco::Pair<Key, Value>> MakePairs(std::size_t distinct, std::size_t multiplicity)
{
    std::vector<aclco::Pair<Key, Value>> pairs;
    pairs.reserve(distinct * multiplicity);
    for (std::size_t i = 0; i < distinct; ++i) {
        for (std::size_t repeat = 0; repeat < multiplicity; ++repeat) {
            pairs.emplace_back(static_cast<Key>(i + 1), static_cast<Value>((i + 1) * 1024 + repeat + 1));
        }
    }
    return pairs;
}

template <typename Key>
std::vector<Key> MakeQueries(std::size_t inserted_distinct, std::size_t query_num, double matching_rate)
{
    const auto hit_num = static_cast<std::size_t>(query_num * matching_rate);
    std::vector<Key> queries;
    queries.reserve(query_num);
    for (std::size_t i = 0; i < hit_num; ++i) {
        queries.push_back(static_cast<Key>((i % std::max<std::size_t>(inserted_distinct, 1)) + 1));
    }
    for (std::size_t i = hit_num; i < query_num; ++i) {
        queries.push_back(static_cast<Key>(inserted_distinct + i + 4096));
    }
    // Interleave hits and misses instead of testing only clustered output.
    for (std::size_t i = 1; i < queries.size(); i += 2) {
        std::swap(queries[i], queries[queries.size() - 1 - i / 2]);
    }
    return queries;
}

template <typename Key, typename Value>
std::unordered_multimap<Key, Value> MapOracle(const std::vector<aclco::Pair<Key, Value>>& pairs)
{
    std::unordered_multimap<Key, Value> oracle;
    oracle.reserve(pairs.size());
    for (auto const& p : pairs) {
        oracle.emplace(p.first, p.second);
    }
    return oracle;
}

template <typename Key>
std::unordered_multiset<Key> SetOracle(const std::vector<Key>& keys)
{
    return std::unordered_multiset<Key>(keys.begin(), keys.end());
}

template <typename Key, typename Oracle>
uint64_t ExpectedCount(Oracle const& oracle, std::vector<Key> const& queries)
{
    uint64_t total = 0;
    for (auto key : queries) {
        total += static_cast<uint64_t>(oracle.count(key));
    }
    return total;
}

template <typename Key, typename Oracle>
std::vector<uint64_t> ExpectedCountEach(Oracle const& oracle, std::vector<Key> const& queries, bool outer = false)
{
    std::vector<uint64_t> expected;
    expected.reserve(queries.size());
    for (auto key : queries) {
        const auto count = static_cast<uint64_t>(oracle.count(key));
        expected.push_back(outer ? std::max<uint64_t>(count, 1) : count);
    }
    return expected;
}

template <typename Key, typename Value>
std::vector<std::pair<Key, Value>> NormalizePairs(std::vector<Key> const& keys, std::vector<Value> const& values,
                                                  std::size_t count)
{
    REQUIRE(keys.size() >= count);
    REQUIRE(values.size() >= count);
    std::vector<std::pair<Key, Value>> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.emplace_back(keys[i], values[i]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

template <typename Key, typename Value>
std::vector<std::pair<Key, Value>> NormalizePairs(std::vector<aclco::Pair<Key, Value>> const& pairs)
{
    std::vector<std::pair<Key, Value>> out;
    out.reserve(pairs.size());
    for (auto const& p : pairs) {
        out.emplace_back(p.first, p.second);
    }
    std::sort(out.begin(), out.end());
    return out;
}

template <typename Key>
std::vector<Key> NormalizeKeys(std::vector<Key> keys, std::size_t count)
{
    REQUIRE(keys.size() >= count);
    keys.resize(count);
    std::sort(keys.begin(), keys.end());
    return keys;
}

template <typename T>
void RequireEqual(std::vector<T> const& actual, std::vector<T> const& expected)
{
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        REQUIRE(actual[i] == expected[i]);
    }
}

} // namespace aclco::test::multi_cases
