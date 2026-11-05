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

TEMPLATE_TEST_CASE_SIG("static_multiset retrieve all correctness", "[static_multiset][retrieveAll]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(5u, 128u, 1027u, 8192u, 100000u);
    auto occupancy = GENERATE(0.0, 0.1, 0.5, 0.9, 1.0);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto key_num = static_cast<std::size_t>(capacity * occupancy);
    auto distinct = key_num == 0u ? 0u : std::max<std::size_t>(1u, key_num / multiplicity);
    CAPTURE(capacity, occupancy, multiplicity, key_num);

    auto set = MakeMultiset<Key>(capacity, stream);
    auto keys = MakeRepeatedKeys<Key>(distinct, multiplicity);
    if (keys.size() > key_num) {
        keys.resize(key_num);
    }
    aclco::test::DeviceBuffer<Key> d_keys(keys.size());
    if (!keys.empty()) {
        d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    }
    set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(keys.size()), stream);

    aclco::test::DeviceBuffer<Key> d_output(keys.size());
    auto retrieved = set.RetrieveAll(d_output.Data(), aclco::Extent<std::size_t>(keys.size()), stream);
    REQUIRE(retrieved == keys.size());
    auto expected = keys;
    std::sort(expected.begin(), expected.end());
    REQUIRE(NormalizeKeys(d_output.CopyToHost(stream), retrieved) == expected);
}

TEMPLATE_TEST_CASE_SIG("static_multiset retrieve_all boundary test", "[static_multiset][retrieveAll][boundary]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto set = MakeMultiset<Key>(128u, sg.stream);
    REQUIRE(set.RetrieveAll(nullptr, aclco::Extent<std::size_t>(0u), sg.stream) == 0u);
}
