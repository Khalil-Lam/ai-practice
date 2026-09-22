# 01｜Tiny Tang Poetry LoRA

这是本仓库的第一个练习项目：用一个非常小的中文语言模型做一次完整的 LoRA 微调。

## 这次真正要学什么

不是“做一个最强唐诗模型”，而是完整走通一次实验链：

```text
公开数据
  ↓
train / dev / test
  ↓
baseline
  ↓
LoRA SFT
  ↓
dev 选择模型
  ↓
held-out test
  ↓
错误分析
```

## 模型

基础模型：

```text
Qwen/Qwen3-0.6B
```

它很小，适合第一次实践。

模型页：

https://huggingface.co/Qwen/Qwen3-0.6B

## 数据

数据来自：

https://github.com/chinese-poetry/chinese-poetry

脚本只读取《全唐诗》的少量 JSON 分片，并默认最多保留 1200 首。

上游仓库 README 标注 MIT License。

本仓库默认不提交生成后的 `train/dev/test.jsonl`，避免把完整数据复制进练习仓库。

## 任务

输入：

```text
请写一首题为《秋夜》的唐诗：
```

目标：

```text
唐诗正文
```

训练时只有“唐诗正文”参与 loss。

prompt 部分的 label 被设为 `-100`。

---

# 第 0 步：进入项目

Windows PowerShell：

```powershell
git clone https://github.com/Khalil-Lam/ai-practice.git
cd ai-practice\01_tang_poetry_lora
```

如果已经 clone 过：

```powershell
git pull
cd 01_tang_poetry_lora
```

---

# 第 1 步：创建环境

```powershell
conda create -n tang-poetry python=3.11 -y
conda activate tang-poetry
```

先安装与你机器匹配的 PyTorch。

官方安装入口：

https://pytorch.org/get-started/locally/

然后：

```powershell
pip install -r requirements.txt
```

检查：

```powershell
python check_env.py
```

你需要重点看：

```text
CUDA available: True
GPU: ...
GPU memory (GB): ...
```

---

# 第 2 步：准备数据

```powershell
python prepare_data.py --files 3 --limit 1200
```

默认：

```text
train = 80%
dev   = 10%
test  = 10%
```

大约会得到：

```text
data/
├─ train.jsonl
├─ dev.jsonl
└─ test.jsonl
```

注意：

**test 不参与训练，不参与选 checkpoint，不参与调参。**

---

# 第 3 步：先跑 baseline

在训练之前先问基础模型：

```powershell
python generate.py --title 秋夜
```

把输出记到 `EXPERIMENT_LOG.md`。

如果不保存 baseline，训练后你就无法判断模型到底学到了什么。

---

# 第 4 步：LoRA 微调

```powershell
python train_lora.py
```

第一版配置：

```text
model = Qwen3-0.6B

LoRA:
r = 8
alpha = 16
dropout = 0.05
target = q_proj, k_proj, v_proj, o_proj

training:
batch size = 1
gradient accumulation = 8
learning rate = 2e-4
epochs = 3
max length = 256
seed = 42
```

checkpoint 按 **dev loss** 选择。

---

# 第 5 步：微调后生成

```powershell
python generate.py --title 秋夜 --adapter outputs/tang-poetry-lora
```

比较：

```text
Baseline
vs.
LoRA
```

先人工看：

1. 是否更像诗；
2. 是否更围绕题目；
3. 五言/七言形式是否更稳定；
4. 是否机械重复；
5. 是否出现背训练集的迹象。

---

# 第 6 步：只在最后看 test

```powershell
python evaluate.py --adapter outputs/tang-poetry-lora --n 20
```

结果写到：

```text
runs/comparison.json
```

这里会同时保留：

- reference
- baseline
- LoRA output
- 简单格式统计
- 是否与训练诗完全相同

这些自动统计只是诊断，不代表“诗歌总质量”。

---

# 第一次实验的成功标准

不要求诗特别好。

满足下面四条就完成第一阶段：

- [ ] 数据能够成功构造；
- [ ] baseline 能正常生成；
- [ ] LoRA 能完整训练并保存 adapter；
- [ ] 微调后的输出和 baseline 出现可解释差异。

---

# 第二轮再做控制实验

第一轮跑通以后，只改一个因素：

```text
Run A: r = 8
Run B: r = 16
```

其他全部保持一致。

这样才能判断变化是不是由 LoRA rank 带来的。

---

## 重要实验纪律

不要：

```text
看 test
→ 改参数
→ 再看 test
→ 再改参数
```

正确流程：

```text
train：学习
dev：调参/选模型
test：最后一次报告
```

---

## Sources

- Qwen3-0.6B:
  https://huggingface.co/Qwen/Qwen3-0.6B
- Hugging Face PEFT / LoRA:
  https://huggingface.co/docs/peft/en/package_reference/lora
- Chinese Poetry dataset:
  https://github.com/chinese-poetry/chinese-poetry
