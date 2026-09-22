import argparse
import json
import random
import re
from pathlib import Path
from urllib.request import urlopen

BASE_URL = (
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
    "%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.{idx}.json"
)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def is_usable(item: dict) -> bool:
    title = clean_text(item.get("title", ""))
    paragraphs = [clean_text(x) for x in item.get("paragraphs", [])]

    if not title or len(paragraphs) < 2 or len(paragraphs) > 8:
        return False

    poem = "".join(paragraphs)
    if len(poem) < 16 or len(poem) > 160:
        return False

    # 初学实验先排除明显带编辑性括号/注记的样本。
    if any(ch in poem for ch in "[]{}<>（）()《》"):
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        type=int,
        default=3,
        help="读取 poet.tang.0/1000/2000... 共多少个分片",
    )
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    rows = []

    for i in range(args.files):
        idx = i * 1000
        url = BASE_URL.format(idx=idx)
        print(f"Downloading: {url}")

        with urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))

        for item in data:
            if not is_usable(item):
                continue

            title = clean_text(item["title"])
            author = clean_text(item.get("author", "佚名"))
            completion = "\n".join(clean_text(x) for x in item["paragraphs"])

            rows.append(
                {
                    "id": item.get("id", ""),
                    "title": title,
                    "author": author,
                    "prompt": f"请写一首题为《{title}》的唐诗：\n",
                    "completion": completion,
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]

    n = len(rows)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)

    splits = {
        "train": rows[:n_train],
        "dev": rows[n_train : n_train + n_dev],
        "test": rows[n_train + n_dev :],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"{split_name}: {len(items)} -> {path}")

    print("\nDone.")
    print("IMPORTANT: test.jsonl 只用于最后评测，不用于选 checkpoint 或调参。")


if __name__ == "__main__":
    main()
