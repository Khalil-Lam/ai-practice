import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_model(adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_poem(title, adapter=None, seed=42):
    torch.manual_seed(seed)

    tokenizer, model = load_model(adapter)

    prompt = f"请写一首题为《{title}》的唐诗：\n"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.08,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="秋夜")
    parser.add_argument(
        "--adapter",
        default=None,
        help="不填=baseline；填 LoRA 路径=微调后模型",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(generate_poem(args.title, args.adapter, args.seed))


if __name__ == "__main__":
    main()
