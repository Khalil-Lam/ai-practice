# AI Practice｜AI 实践练习

这是我的 AI 实践仓库，用来保存可复现的小实验。

目标不是堆项目，而是把每个实验完整走通：

```text
问题 -> 数据 -> baseline -> 训练/微调 -> 验证 -> 测试 -> 错误分析 -> 结论
```

## Projects

### 01. Tiny Tang Poetry LoRA

用一个很小的中文语言模型做第一次 LoRA 微调练习：

- 基础模型：Qwen3-0.6B
- 数据：公开《全唐诗》数据
- 任务：给定诗题，生成唐诗
- 方法：SFT + LoRA
- 重点：严格分开 train / dev / test，并比较微调前后的输出

目录：[`01_tang_poetry_lora/`](./01_tang_poetry_lora/)

---

每个项目都保留数据来源、随机种子、训练配置、验证方法和失败记录。
