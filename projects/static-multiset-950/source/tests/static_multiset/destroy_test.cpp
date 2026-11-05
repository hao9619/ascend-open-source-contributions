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

TEMPLATE_TEST_CASE_SIG("static_multiset destroy lifecycle", "[static_multiset][destroy]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(5u, 128u, 8192u);
    auto occupancy = GENERATE(0.0, 0.5, 0.9);
    CAPTURE(capacity, occupancy);

    for (int round = 0; round < 8; ++round) {
        auto set = MakeMultiset<Key>(capacity, stream);
        auto key_num = static_cast<std::size_t>(capacity * occupancy);
        auto keys = MakeRepeatedKeys<Key>(key_num, 1u);
        aclco::test::DeviceBuffer<Key> d_keys(keys.size());
        if (!keys.empty()) {
            d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
        }
        REQUIRE(set.Insert(d_keys.Data(), aclco::Extent<std::size_t>(keys.size()), stream) == 0u);
        REQUIRE(set.Size(stream) == keys.size());
    }
}
