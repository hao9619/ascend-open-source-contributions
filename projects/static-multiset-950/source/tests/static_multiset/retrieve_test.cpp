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

TEMPLATE_TEST_CASE_SIG("static_multiset retrieve correctness", "[static_multiset][retrieve]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    RunQueryCase<Key, 8192u, 1024u>([](auto& set, auto& oracle, auto& queries, auto stream) {
        auto expected_count = ExpectedCount(oracle, queries);
        aclco::test::DeviceBuffer<Key> d_queries(queries.size());
        aclco::test::DeviceBuffer<Key> d_probe_out(expected_count);
        aclco::test::DeviceBuffer<Key> d_match_out(expected_count);
        d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);

        auto retrieved = set.Retrieve(d_queries.Data(), aclco::Extent<std::size_t>(queries.size()), d_probe_out.Data(),
                                      d_match_out.Data(), aclco::Extent<std::size_t>(expected_count), stream);
        REQUIRE(retrieved == expected_count);

        auto probes = d_probe_out.CopyToHost(stream);
        auto matches = d_match_out.CopyToHost(stream);
        REQUIRE(probes.size() >= retrieved);
        REQUIRE(matches.size() >= retrieved);
        for (std::size_t i = 0; i < retrieved; ++i) {
            REQUIRE(probes[i] == matches[i]);
            REQUIRE(oracle.count(matches[i]) != 0u);
        }
    });
}

TEMPLATE_TEST_CASE_SIG("static_multiset retrieve boundary test", "[static_multiset][retrieve][boundary]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto set = MakeMultiset<Key>(128u, sg.stream);
    REQUIRE(set.Retrieve(nullptr, aclco::Extent<std::size_t>(0u), nullptr, nullptr, aclco::Extent<std::size_t>(0u),
                         sg.stream) == 0u);
}
