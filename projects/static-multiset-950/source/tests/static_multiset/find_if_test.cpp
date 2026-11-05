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

TEMPLATE_TEST_CASE_SIG("static_multiset find_if correctness", "[static_multiset][find_if]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    RunConditionalCase<Key>([](auto& set, auto& queries, auto& stencil, auto count, auto capacity, auto matching_rate,
                               auto stencil_mode, auto stream) {
        aclco::test::DeviceBuffer<Key> d_queries(count);
        aclco::test::DeviceBuffer<uint32_t> d_stencil(count);
        aclco::test::DeviceBuffer<Key> d_output(count);
        d_queries.CopyFromHostAsync(queries.data(), count, stream);
        d_stencil.CopyFromHostAsync(stencil.data(), count, stream);
        d_output.MemsetZero(stream);
        set.template FindIf<uint32_t, IsOdd>(static_cast<void*>(d_queries.Data()), d_stencil.Data(),
                                             static_cast<void*>(d_output.Data()), aclco::Extent<std::size_t>(count),
                                             stream);
        auto actual = d_output.CopyToHost(stream);
        for (std::size_t i = 0; i < count; ++i) {
            auto expected = stencil[i] % 2u != 0u && queries[i] <= Key(count) ? queries[i] : EmptyKey<Key>();
            CAPTURE(i, capacity, matching_rate, stencil_mode);
            REQUIRE(actual[i] == expected);
        }
    });
}

TEMPLATE_TEST_CASE_SIG("static_multiset find_if boundary test", "[static_multiset][find_if][boundary]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto set = MakeMultiset<Key>(128u, sg.stream);
    set.template FindIf<uint32_t, IsOdd>(nullptr, nullptr, nullptr, aclco::Extent<std::size_t>(0u), sg.stream);
    REQUIRE(set.Size(sg.stream) == 0u);
}
