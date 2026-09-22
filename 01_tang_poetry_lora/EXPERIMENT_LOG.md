# Experiment Log

## Run 001

### Environment

- Date: 2026-09-22
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- GPU memory: 7.96 GB
- Python: 3.11.15
- PyTorch: 2.11.0+cu128
- Transformers: 5.17.0
- PEFT: 0.21.0
- CUDA available: True
- BF16 supported: True

### Data

- Upstream source:
- Data preparation command:
- Train count:
- Dev count:
- Test count:
- Seed: 42

### Model

- Base model: Qwen/Qwen3-0.6B
- LoRA r: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj

### Training

- Learning rate: 2e-4
- Epochs: 3
- Batch size: 1
- Gradient accumulation: 8
- Max length: 256
- Best dev loss:
- Runtime:

### Baseline example

Title:

Output:

### LoRA example

Title:

Output:

### Observations

- What improved?
- What became worse?
- Any repetition?
- Any obvious memorization?
- Any malformed output?

### Decision

- Continue / stop:
- Next experiment:
- Why:
