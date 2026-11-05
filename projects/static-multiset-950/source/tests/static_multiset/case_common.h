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
#include "common/multi_container_test_common.h"
#include <tuple>

namespace aclco::test::multi_cases {

inline auto CountCaseParameters()
{
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto requestedMultiplicity = GENERATE(1u, 2u, 4u, 8u, 1024u);
    auto multiplicity = std::min<std::size_t>(requestedMultiplicity, capacity / 2u);
    auto matching_rate = GENERATE(0.0, 0.1, 0.5, 1.0);
    auto distinct = std::max<std::size_t>(1u, capacity / (2u * multiplicity));
    return std::make_tuple(capacity, requestedMultiplicity, multiplicity, matching_rate, distinct);
}

template <unsigned MaxCapacity>
auto QueryCaseParameters()
{
    auto capacity = GENERATE(128u, 1027u, MaxCapacity);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto matching_rate = GENERATE(0.0, 0.1, 0.5, 1.0);
    auto distinct = std::max<std::size_t>(1u, capacity / (2u * multiplicity));
    return std::make_tuple(capacity, multiplicity, matching_rate, distinct);
}

inline auto ConditionalCaseParameters()
{
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto matching_rate = GENERATE(0.0, 0.5, 1.0);
    auto stencil_mode = GENERATE(0u, 1u, 2u);
    auto count = std::min<std::size_t>(capacity / 2u, 4096u);
    return std::make_tuple(capacity, matching_rate, stencil_mode, count);
}

template <typename Key>
void InsertTestKeys(aclco::StaticMultiset<Key>& set, const std::vector<Key>& keys, DeviceBuffer<Key>& deviceKeys,
                    aclrtStream stream)
{
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    set.Insert(deviceKeys.Data(), aclco::Extent<std::size_t>(keys.size()), stream);
}

template <typename Key>
struct PreparedCase {
    aclco::StaticMultiset<Key> set;
    std::vector<Key> keys;
    std::unordered_multiset<Key> oracle;
    DeviceBuffer<Key> deviceKeys;

    PreparedCase(std::size_t capacity, std::size_t distinct, std::size_t multiplicity, aclrtStream stream)
        : set(MakeMultiset<Key>(capacity, stream)),
          keys(MakeRepeatedKeys<Key>(distinct, multiplicity)),
          oracle(SetOracle(keys)),
          deviceKeys(keys.size())
    {
        InsertTestKeys(set, keys, deviceKeys, stream);
    }
};

inline std::vector<uint32_t> MakeStencil(std::size_t count, unsigned stencil_mode)
{
    std::vector<uint32_t> stencil(count);
    for (std::size_t i = 0; i < count; ++i) {
        stencil[i] = stencil_mode == 0u ? 0u : (stencil_mode == 1u ? 1u : static_cast<uint32_t>(i));
    }
    return stencil;
}

// Keep GENERATE and CAPTURE inside the active test invocation. Each caller is
// still a separately registered Catch2 test with its original assertion body.
template <typename Key, typename Check>
void RunCountCase(Check check)
{
    AclGlobalGuard g_acl;
    AclStreamGuard sg;
    auto stream = sg.stream;
    auto [capacity, requestedMultiplicity, multiplicity, matching_rate, distinct] = CountCaseParameters();
    CAPTURE(capacity, requestedMultiplicity, multiplicity, matching_rate);
    auto [set, keys, oracle, deviceKeys] = PreparedCase<Key>(capacity, distinct, multiplicity, stream);
    auto queries = MakeQueries<Key>(distinct, std::min<std::size_t>(4096u, capacity), matching_rate);
    check(set, oracle, queries, stream, distinct);
}

template <typename Key, unsigned MaxCapacity, unsigned MaxQueries, typename Check>
void RunQueryCase(Check check)
{
    AclGlobalGuard g_acl;
    AclStreamGuard sg;
    auto stream = sg.stream;
    auto [capacity, multiplicity, matching_rate, distinct] = QueryCaseParameters<MaxCapacity>();
    CAPTURE(capacity, multiplicity, matching_rate);
    auto [set, keys, oracle, deviceKeys] = PreparedCase<Key>(capacity, distinct, multiplicity, stream);
    auto queries = MakeQueries<Key>(distinct, std::min<std::size_t>(MaxQueries, capacity), matching_rate);
    check(set, oracle, queries, stream);
}

template <typename Key, typename Check>
void RunConditionalCase(Check check)
{
    AclGlobalGuard g_acl;
    AclStreamGuard sg;
    auto stream = sg.stream;
    auto [capacity, matching_rate, stencil_mode, count] = ConditionalCaseParameters();
    auto set = MakeMultiset<Key>(capacity, stream);
    auto keys = MakeRepeatedKeys<Key>(count, 1u);
    DeviceBuffer<Key> deviceKeys(count);
    InsertTestKeys(set, keys, deviceKeys, stream);
    auto queries = MakeQueries<Key>(count, count, matching_rate);
    auto stencil = MakeStencil(count, stencil_mode);
    check(set, queries, stencil, count, capacity, matching_rate, stencil_mode, stream);
}

struct IsOdd {
    COLLECTION_SIMT_DEVICE bool operator()(uint32_t value) const noexcept { return value % 2u != 0u; }
};
} // namespace aclco::test::multi_cases
