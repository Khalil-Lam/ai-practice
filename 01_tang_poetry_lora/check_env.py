import sys

import torch
import transformers
import peft


print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("GPU memory (GB):", round(props.total_memory / 1024**3, 2))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
else:
    print("WARNING: CUDA is not available. Training will be very slow on CPU.")
