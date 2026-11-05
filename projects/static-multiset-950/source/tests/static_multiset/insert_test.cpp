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
#include "common/multi_container_test_common.h"

using namespace aclco::test::multi_cases;

TEMPLATE_TEST_CASE_SIG("static_multiset insert correctness", "[static_multiset][insert]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(128u, 1027u, 8192u, 65536u);
    auto occupancy = GENERATE(0.1, 0.5, 0.9);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto key_num = static_cast<std::size_t>(capacity * occupancy);
    auto distinct = std::max<std::size_t>(key_num / multiplicity, 1u);
    CAPTURE(capacity, occupancy, multiplicity, key_num);

    auto set = MakeMultiset<Key>(capacity, stream);
    auto keys = MakeRepeatedKeys<Key>(distinct, multiplicity);
    if (keys.size() > key_num) {
        keys.resize(key_num);
    }
    aclco::test::DeviceBuffer<Key> d_keys(keys.size());
    d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    REQUIRE(set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(keys.size()), stream) == 0u);
    REQUIRE(set.Size(stream) == keys.size());

    SECTION("duplicate key is retained")
    {
        std::vector<Key> repeats(8u, keys.front());
        aclco::test::DeviceBuffer<Key> d_repeats(repeats.size());
        d_repeats.CopyFromHostAsync(repeats.data(), repeats.size(), stream);
        auto before = set.Size(stream);
        REQUIRE(set.Insert(d_repeats.Data(), aclco::Extent<std::size_t>(repeats.size()), stream) == 0u);
        REQUIRE(set.Size(stream) == before + repeats.size());
    }

    SECTION("zero and null inputs")
    {
        auto before = set.Size(stream);
        REQUIRE(set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(0u), stream) == 0u);
        REQUIRE(set.Size(stream) == before);
        REQUIRE(set.Insert(nullptr, aclco::Extent<std::size_t>(3u), stream) == 3u);
    }
}

TEMPLATE_TEST_CASE_SIG("static_multiset insert capacity boundary", "[static_multiset][insert][boundary]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto set = MakeMultiset<Key>(128u, stream);
    auto capacity = set.Capacity();
    auto keys = MakeRepeatedKeys<Key>(capacity + 5u, 1u);
    aclco::test::DeviceBuffer<Key> d_keys(keys.size());
    d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    auto failures = set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(keys.size()), stream);
    REQUIRE(failures == keys.size() - capacity);
    REQUIRE(set.Size(stream) == capacity);
}
