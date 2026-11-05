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

TEMPLATE_TEST_CASE_SIG("static_multiset count each outer correctness", "[static_multiset][countEachOuter]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    RunCountCase<Key>([](auto& set, auto& oracle, auto& queries, auto stream, auto distinct) {
        queries.insert(queries.begin(), {static_cast<Key>(1), static_cast<Key>(distinct + 9999)});
        aclco::test::DeviceBuffer<Key> d_queries(queries.size());
        aclco::test::DeviceBuffer<uint64_t> d_output(queries.size());
        d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
        set.CountEachOuter(d_queries.Data(), d_output.Data(), aclco::Extent<std::size_t>(queries.size()), stream);

        RequireEqual(d_output.CopyToHost(stream), ExpectedCountEach(oracle, queries, true));
    });
}

TEMPLATE_TEST_CASE_SIG("static_multiset count_each_outer boundary test", "[static_multiset][countEachOuter][boundary]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto set = MakeMultiset<Key>(128u, sg.stream);
    set.CountEachOuter(nullptr, nullptr, aclco::Extent<std::size_t>(0u), sg.stream);
    REQUIRE(set.Size(sg.stream) == 0u);
}
