# 贡献范围与来源

本仓展示我为华为昇腾 Atlas 950PR 开发的 `aclblasCgemm3m` 接口、三乘法计算路径和配套测试。开发中复用了已有 Cgemm 组件。新增与修改的具体范围见 [cann/ops-blas PR #735](https://gitcode.com/cann/ops-blas/pull/735)。

## 源码版本

- **代码来源**　`ops-blas/` 对应合入时的 PR 源提交 `a2626f23ec202b1f8bdd3d637cae8dbf6b7decb8`。收录该提交相对父提交的全部 26 个变更文件，另附两份供阅读使用的上游共享头文件 `cgemm_onchip_kernel.h`、`cgemm_kernel_common.h`，以及原始许可证。[完整源码树](https://gitcode.com/gzh1630/ops-blas/tree/a2626f23ec202b1f8bdd3d637cae8dbf6b7decb8)
- **许可**　`ops-blas/` 内的文件保留上游版权和声明，适用随附的 CANN Open Software License Agreement Version 2.0。仓库根目录的 MIT 许可证适用于本仓原创展示内容。[上游许可证](ops-blas/LICENSE) · [展示仓许可证](LICENSE)
- **提交记录**　使用朋友提供的 GitCode 账号 `gzh1630` 提交，于 2026 年 11 月 21 日合入。开发过程中使用了 OpenAI Codex，参与范围见上游 PR 正文。

## 图片来源

首页的矩阵插画由 AI 生成。项目背景旁的标识取自 Huawei Ascend 官方 GitHub 组织头像，链接指向该组织，品牌归原权利人所有。[Huawei Ascend](https://github.com/Ascend)
