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

TEMPLATE_TEST_CASE_SIG("static_multiset find correctness", "[static_multiset][find]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    RunQueryCase<Key, 100000u, 4096u>([](auto& set, auto& oracle, auto& queries, auto stream) {
        aclco::test::DeviceBuffer<Key> d_queries(queries.size());
        aclco::test::DeviceBuffer<Key> d_output(queries.size());
        d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
        set.Find(d_queries.Data(), d_output.Data(), aclco::Extent<std::size_t>(queries.size()), stream);

        auto actual = d_output.CopyToHost(stream);
        for (std::size_t i = 0; i < queries.size(); ++i) {
            auto expected = oracle.count(queries[i]) == 0u ? EmptyKey<Key>() : queries[i];
            CAPTURE(i, queries[i]);
            REQUIRE(actual[i] == expected);
        }
    });
}

TEMPLATE_TEST_CASE_SIG("static_multiset find negative test", "[static_multiset][find][negative]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto set = MakeMultiset<Key>(128u, stream);
    auto keys = MakeRepeatedKeys<Key>(5u, 2u);
    aclco::test::DeviceBuffer<Key> d_keys(keys.size());
    d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(keys.size()), stream);

    SECTION("duplicate queries return the queried key")
    {
        std::vector<Key> queries(7u, static_cast<Key>(1));
        aclco::test::DeviceBuffer<Key> d_queries(queries.size());
        aclco::test::DeviceBuffer<Key> d_output(queries.size());
        d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
        set.Find(d_queries.Data(), d_output.Data(), aclco::Extent<std::size_t>(queries.size()), stream);
        for (auto key : d_output.CopyToHost(stream)) {
            REQUIRE(key == static_cast<Key>(1));
        }
    }

    SECTION("zero extent accepts null buffers")
    {
        set.Find(nullptr, nullptr, aclco::Extent<std::size_t>(0u), stream);
        REQUIRE(set.Size(stream) == keys.size());
    }
}
