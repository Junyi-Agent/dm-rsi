---
name: 契约变更
about: 你的 sub-RSI 要改输入或输出文件的格式（列、文件名、路径约定、CLI）
title: "[契约] "
labels: []
---

**方向**：Data / Training / Decision Policy

**现在的契约**：例如 `<name>_{pool,val,test}.parquet`，列 `id, y, p[, step, post43]`。

**改成**：

**什么时候生效**：sub-RSI 的哪个版本 / commit。dm-rsi 会在升级 submodule 指针时同步改适配器。
