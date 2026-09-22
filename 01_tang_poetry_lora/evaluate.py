import argparse
import json
import re
from pathlib import Path

from generate import generate_poem


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text):
    return re.sub(r"\s+", "", text)


def segment_lengths(text):
    segments = [
        x.strip()
        for x in re.split(r"[。！？!?]", text)
        if x.strip()
    ]

    lengths = []
    for segment in segments:
        plain = re.sub(
            r"[，、；：,.!?！？。；：]",
            "",
            segment,
        )
        if plain:
            lengths.append(len(plain))

    return lengths


def simple_stats(text, train_poems):
    normalized = normalize(text)
    lengths = segment_lengths(text)

    if lengths:
        regular_rate = sum(
            length in (5, 7, 10, 14)
            for length in lengths
        ) / len(lengths)
    else:
        regular_rate = 0.0

    return {
        "characters": len(normalized),
        "segments": len(lengths),
        "regular_segment_rate": round(regular_rate, 3),
        "exact_train_match": normalized in train_poems,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="outputs/tang-poetry-lora",
    )
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--out",
        default="runs/comparison.json",
    )
    args = parser.parse_args()

    test_rows = read_jsonl(args.test)[: args.n]
    train_rows = read_jsonl(args.train)

    train_poems = {
        normalize(row["completion"])
        for row in train_rows
    }

    results = []

    for i, item in enumerate(test_rows):
        title = item["title"]
        seed = 42 + i

        print(f"[{i + 1}/{len(test_rows)}] {title}")

        baseline = generate_poem(
            title,
            adapter=None,
            seed=seed,
        )
        lora_output = generate_poem(
            title,
            adapter=args.adapter,
            seed=seed,
        )

        results.append(
            {
                "title": title,
                "reference": item["completion"],
                "baseline": baseline,
                "lora": lora_output,
                "baseline_stats": simple_stats(
                    baseline,
                    train_poems,
                ),
                "lora_stats": simple_stats(
                    lora_output,
                    train_poems,
                ),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved comparison -> {out_path}")
    print(
        "NOTE: 这些格式统计只是诊断，不是诗歌质量的最终评分。"
    )


if __name__ == "__main__":
    main()
