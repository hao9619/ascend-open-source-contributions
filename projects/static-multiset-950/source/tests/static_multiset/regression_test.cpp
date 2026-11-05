/* Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root. */
#include "common/multi_container_test_common.h"
using namespace aclco::test::multi_cases;
using namespace aclco::test;

namespace {
template <class K>
std::vector<K> MakeDistributionKeys(uint32_t n, uint32_t pattern)
{
    std::vector<K> keys(n);
    uint64_t seed = 0x9e3779b97f4a7c15ULL;
    for (uint32_t i = 0; i < n; ++i) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t x = seed;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x ^= x >> 31;
        if (pattern == 0)
            keys[i] = static_cast<K>(x);
        if (pattern == 1)
            keys[i] = static_cast<K>(uint64_t(i) * 32 + 1);
        if (pattern == 2)
            keys[i] = static_cast<K>((uint64_t(i % 37) << 32) | (i % 37 + 1));
        if (pattern == 3)
            keys[i] = static_cast<K>(uint64_t(i) + 123);
        if (keys[i] == EmptyKey<K>())
            keys[i] = 7;
        if (pattern != 2 && i && i % 7 == 0)
            keys[i] = keys[i - 1];
    }
    return keys;
}

template <class K>
COLLECTION_SIMT_VF LAUNCH_BOUND(32) inline void QueryRefSimt(aclco::StaticMultisetRef<K> ref, __gm__ K* keys,
                                                             __gm__ uint64_t* output)
{
    uint32_t i = AscendC::Simt::GetThreadIdx();
    if (i < 4)
        output[i] = ref.Count(keys[i]);
}
template <class K>
COLLECTION_AIV_GLOBAL void QueryRefKernel(aclco::StaticMultisetRef<K> ref, __gm__ K* keys, __gm__ uint64_t* output)
{
    AscendC::Simt::VF_CALL<QueryRefSimt<K>>(AscendC::Simt::Dim3{32}, ref, keys, output);
}
} // namespace

TEMPLATE_TEST_CASE("static_multiset stable results and failure boundaries", "[static_multiset][regression]", int32_t,
                   int64_t)
{
    using K = TestType;
    using E = aclco::Extent<std::size_t>;
    AclGlobalGuard acl;
    AclStreamGuard sg;
    auto stream = sg.stream;
    std::vector<K> values{5, -2, 5, EmptyKey<K>(), 7, -2, 9, 11};
    DeviceBuffer<K> input(values.size());
    input.CopyFromHostAsync(values.data(), values.size(), stream);
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto set = MakeMultiset<K>(6, stream);
        REQUIRE(set.Insert(input.Data(), E(values.size()), stream) == 2);
        REQUIRE(set.Size(stream) == 6);
        auto ref = set.Ref();
        REQUIRE(ref.slots >= set.Capacity());
        REQUIRE_THROWS_AS(set.Count(input.Data(), E(std::numeric_limits<std::size_t>::max()), stream),
                          std::overflow_error);
        DeviceBuffer<K> all(6);
        REQUIRE(set.RetrieveAll(all.Data(), E(6), stream) == 6);
        REQUIRE(all.CopyToHost(stream) == std::vector<K>{5, -2, 5, 7, -2, 9});
        REQUIRE_THROWS_AS(set.RetrieveAll(all.Data(), E(5), stream), std::length_error);
        std::vector<K> queries{-2, 99, 5, -2};
        DeviceBuffer<K> query(4), probes(6), matches(6);
        query.CopyFromHostAsync(queries.data(), queries.size(), stream);
        probes.MemsetZero(stream);
        REQUIRE_THROWS_AS(set.Retrieve(query.Data(), E(4), probes.Data(), matches.Data(), E(5), stream),
                          std::length_error);
        REQUIRE(probes.CopyToHost(stream) == std::vector<K>(6, 0));
        REQUIRE(set.Retrieve(query.Data(), E(4), probes.Data(), matches.Data(), E(6), stream) == 6);
        auto expected = std::vector<K>{-2, -2, 5, 5, -2, -2};
        REQUIRE(probes.CopyToHost(stream) == expected);
        REQUIRE(matches.CopyToHost(stream) == expected);
        auto moved = std::move(set);
        REQUIRE_THROWS_AS(set.Size(stream), std::logic_error);
        REQUIRE(moved.Size(stream) == 6);
        moved.Destroy();
        moved.Destroy();
        REQUIRE_THROWS_AS(moved.Clear(stream), std::logic_error);
    }
}

TEMPLATE_TEST_CASE("static_multiset scan across tile levels", "[static_multiset][regression]", int32_t, int64_t)
{
    using K = TestType;
    using E = aclco::Extent<std::size_t>;
    AclGlobalGuard acl;
    AclStreamGuard sg;
    auto set = MakeMultiset<K>(100, sg.stream);
    std::vector<K> keys{17, 17, 17, 42};
    DeviceBuffer<K> input(4);
    input.CopyFromHostAsync(keys.data(), 4, sg.stream);
    REQUIRE(set.Insert(input.Data(), E(4), sg.stream) == 0);
    auto n = GENERATE(255u, 256u, 257u, 65537u);
    std::vector<K> q(n), expected;
    for (std::size_t i = 0; i < n; ++i) {
        q[i] = i % 3 == 0 ? 17 : (i % 3 == 1 ? 42 : 99);
        if (q[i] == 17)
            expected.insert(expected.end(), 3, K(17));
        if (q[i] == 42)
            expected.push_back(42);
    }
    DeviceBuffer<K> queries(n), probes(expected.size()), matches(expected.size());
    queries.CopyFromHostAsync(q.data(), n, sg.stream);
    REQUIRE(set.Count(queries.Data(), E(n), sg.stream) == expected.size());
    REQUIRE(set.Retrieve(queries.Data(), E(n), probes.Data(), matches.Data(), E(expected.size()), sg.stream) ==
            expected.size());
    REQUIRE(probes.CopyToHost(sg.stream) == expected);
    REQUIRE(matches.CopyToHost(sg.stream) == expected);
}

TEMPLATE_TEST_CASE("static_multiset high multiplicity and wide keys", "[static_multiset][regression]", int32_t, int64_t)
{
    using K = TestType;
    using E = aclco::Extent<std::size_t>;
    AclGlobalGuard acl;
    AclStreamGuard sg;
    aclco::StaticMultiset<K> set(E(8192), K(0), sg.stream);
    K high = std::numeric_limits<K>::max() - 5;
    K low = std::numeric_limits<K>::lowest() + 1;
    std::vector<K> keys(4096, high);
    keys.insert(keys.end(), 2048, low);
    DeviceBuffer<K> input(keys.size());
    input.CopyFromHostAsync(keys.data(), keys.size(), sg.stream);
    REQUIRE(set.Insert(input.Data(), E(keys.size()), sg.stream) == 0);
    REQUIRE(set.Size(sg.stream) == keys.size());
    K hostQueries[] = {high, low, 0, K(high - 1)};
    DeviceBuffer<K> queries(4);
    DeviceBuffer<uint64_t> counts(4);
    queries.CopyFromHostAsync(hostQueries, 4, sg.stream);
    set.CountEach(queries.Data(), counts.Data(), E(4), sg.stream);
    REQUIRE(counts.CopyToHost(sg.stream) == std::vector<uint64_t>{4096, 2048, 0, 0});
    REQUIRE(set.Count(queries.Data(), E(4), sg.stream) == keys.size());
    counts.MemsetZero(sg.stream);
    QueryRefKernel<K><<<1, 0, sg.stream>>>(set.Ref(), queries.Data(), counts.Data());
    REQUIRE(counts.CopyToHost(sg.stream) == std::vector<uint64_t>{4096, 2048, 0, 0});
}

TEMPLATE_TEST_CASE("static_multiset key distributions and warp tails", "[static_multiset][regression]", int32_t,
                   int64_t)
{
    using K = TestType;
    using E = aclco::Extent<std::size_t>;
    AclGlobalGuard acl;
    AclStreamGuard sg;
    auto n = GENERATE(31u, 32u, 33u, 1023u, 1025u, 65537u);
    auto pattern = GENERATE(0u, 1u, 2u, 3u);
    CAPTURE(n, pattern);
    auto keys = MakeDistributionKeys<K>(n, pattern);
    auto sorted = keys;
    std::sort(sorted.begin(), sorted.end());
    std::vector<K> queries = keys;
    queries.push_back(EmptyKey<K>());
    std::vector<uint64_t> expected(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        auto range = std::equal_range(sorted.begin(), sorted.end(), queries[i]);
        expected[i] = static_cast<uint64_t>(range.second - range.first);
    }
    DeviceBuffer<K> input(n), q(queries.size()), found(queries.size()), all(n);
    DeviceBuffer<uint64_t> counts(queries.size());
    DeviceBuffer<uint8_t> contains(queries.size());
    input.CopyFromHostAsync(keys.data(), n, sg.stream);
    q.CopyFromHostAsync(queries.data(), queries.size(), sg.stream);
    auto set = MakeMultiset<K>(n + 8, sg.stream);
    auto first = n / 3;
    REQUIRE(set.Insert(input.Data(), E(first), sg.stream) == 0);
    REQUIRE(set.Insert(input.Data() + first, E(n - first), sg.stream) == 0);
    REQUIRE(set.Size(sg.stream) == n);
    set.CountEach(q.Data(), counts.Data(), E(queries.size()), sg.stream);
    REQUIRE(counts.CopyToHost(sg.stream) == expected);
    set.Contains(q.Data(), contains.Data(), E(queries.size()), sg.stream);
    auto flags = contains.CopyToHost(sg.stream);
    for (size_t i = 0; i < flags.size(); ++i)
        REQUIRE(bool(flags[i]) == (expected[i] != 0));
    set.Find(q.Data(), found.Data(), E(queries.size()), sg.stream);
    REQUIRE(found.CopyToHost(sg.stream) == queries);
    REQUIRE(set.RetrieveAll(all.Data(), E(n), sg.stream) == n);
    REQUIRE(all.CopyToHost(sg.stream) == keys);
    set.Clear(sg.stream);
    REQUIRE(set.Count(q.Data(), E(queries.size()), sg.stream) == 0);
    REQUIRE(set.Insert(input.Data(), E(n), sg.stream) == 0);
    set.CountEach(q.Data(), counts.Data(), E(queries.size()), sg.stream);
    REQUIRE(counts.CopyToHost(sg.stream) == expected);
}
