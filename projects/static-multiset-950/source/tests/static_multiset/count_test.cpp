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
#include "case_common.h"

using namespace aclco::test::multi_cases;

TEMPLATE_TEST_CASE_SIG("static_multiset count correctness", "[static_multiset][count]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    RunCountCase<Key>([](auto& set, auto& oracle, auto& queries, auto stream, auto) {
        aclco::test::DeviceBuffer<Key> d_queries(queries.size());
        d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
        auto actual = set.Count(d_queries.Data(), aclco::Extent<std::size_t>(queries.size()), stream);
        REQUIRE(actual == ExpectedCount(oracle, queries));
    });
}

TEMPLATE_TEST_CASE_SIG("static_multiset count negative test", "[static_multiset][count][negative]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    constexpr std::size_t multiplicity = 4u;
    auto set = MakeMultiset<Key>(128u, stream);
    auto keys = MakeRepeatedKeys<Key>(5u, multiplicity);
    aclco::test::DeviceBuffer<Key> d_keys(keys.size());
    d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(keys.size()), stream);

    SECTION("duplicate queries preserve multiplicity")
    {
        std::vector<Key> queries(7u, static_cast<Key>(1));
        aclco::test::DeviceBuffer<Key> d_queries(queries.size());
        d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
        REQUIRE(set.Count(d_queries.Data(), aclco::Extent<std::size_t>(queries.size()), stream) ==
                queries.size() * multiplicity);
    }

    SECTION("zero extent accepts null input")
    {
        REQUIRE(set.Count(nullptr, aclco::Extent<std::size_t>(0u), stream) == 0u);
    }
}
