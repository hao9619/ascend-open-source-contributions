/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
 */
#include <gtest/gtest.h>

#include "event_report.h"
#include "nlohmann/json.hpp"

using namespace MemScope;
using Json = nlohmann::json;

namespace
{
Json Message()
{
    return {{"name", "copy_kernel"},
            {"phase", "begin"},
            {"device", 0},
            {"call_id", 1},
            {"stream_handle", 0},
            {"extent_kind", "parameter_storage_span"},
            {"effect_kind", "static_possible"},
            {"accesses", Json::array({{{"parameter", "x"}, {"ptr", 4096}, {"size", 16},
                                      {"read", true}, {"write", false}, {"unknown", false}}})}};
}

bool Report(const Json& message)
{
    return EventReport::Instance(MemScopeCommType::MEMORY_DEBUG).ReportTritonMessage(message.dump().c_str());
}
}  // namespace

TEST(TritonReportTest, RejectsNullAndMalformed)
{
    auto& report = EventReport::Instance(MemScopeCommType::MEMORY_DEBUG);
    EXPECT_FALSE(report.ReportTritonMessage(nullptr));
    EXPECT_FALSE(report.ReportTritonMessage("{bad json"));
    EXPECT_FALSE(report.ReportTritonMessage("{}"));
    const std::string oversized(1024 * 1024 + 1, 'x');
    EXPECT_FALSE(report.ReportTritonMessage(oversized.c_str()));
}

TEST(TritonReportTest, RejectsInvalidRanges)
{
    auto message = Message();
    for (const auto size : {Json(0), Json(-1), Json(1.5), Json("16"), Json(uint64_t(1) << 63)})
    {
        message["accesses"][0]["size"] = size;
        EXPECT_FALSE(Report(message));
    }
    message = Message();
    message["accesses"][0]["ptr"] = UINT64_MAX - 1;
    EXPECT_FALSE(Report(message));
    message["accesses"][0]["ptr"] = 0;
    EXPECT_FALSE(Report(message));
}

TEST(TritonReportTest, RejectsInvalidMetadata)
{
    for (const auto key : {"phase", "device", "call_id", "stream_handle", "effect_kind", "extent_kind"})
    {
        auto message = Message();
        message[key] = -1;
        EXPECT_FALSE(Report(message));
    }
    auto message = Message();
    message["accesses"][0]["write"] = "false";
    EXPECT_FALSE(Report(message));
    message = Message();
    message["name"] = "copy,spoofed\nrow";
    EXPECT_FALSE(Report(message));
}

TEST(TritonReportTest, ValidatesAllParametersBeforeEmission)
{
    auto message = Message();
    message["accesses"].push_back({{"parameter", "bad"}});
    EXPECT_FALSE(Report(message));
    message["accesses"] = Json::object();
    EXPECT_FALSE(Report(message));
}
