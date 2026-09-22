# Experiment Log

## Run 001

### Environment

- Date:
- GPU:
- GPU memory:
- Python:
- PyTorch:
- Transformers:
- PEFT:

### Data

- Upstream source:
- Data preparation command:
- Train count:
- Dev count:
- Test count:
- Seed:

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
