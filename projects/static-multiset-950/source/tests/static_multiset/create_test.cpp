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

TEMPLATE_TEST_CASE_SIG("static_multiset create correctness", "[static_multiset][create]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto requested = GENERATE(1u, 5u, 128u, 1027u, 8192u, 1000000u);
    CAPTURE(requested);

    auto set = MakeMultiset<Key>(requested, stream);
    REQUIRE(set.Capacity() >= requested);
    REQUIRE(set.Size(stream) == 0u);

    aclco::test::DeviceBuffer<Key> d_output(set.Capacity());
    REQUIRE(set.RetrieveAll(d_output.Data(), aclco::Extent<std::size_t>(set.Capacity()), stream) == 0u);
}

TEMPLATE_TEST_CASE_SIG("static_multiset create invalid capacity", "[static_multiset][create][negative]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    REQUIRE_THROWS(MakeMultiset<Key>(0u, sg.stream));
}
