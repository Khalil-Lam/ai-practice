import json
from pathlib import Path

REQUIRED = {"id", "title", "author", "prompt", "completion"}
SPLITS = ["train", "dev", "test"]


def load_split(name):
    path = Path("data") / f"{name}.jsonl"
    rows = []
    bad_json = 0
    missing = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            if not REQUIRED.issubset(item):
                missing += 1

            rows.append(item)

    return rows, bad_json, missing


def main():
    loaded = {}
    for split in SPLITS:
        rows, bad_json, missing = load_split(split)
        loaded[split] = rows
        print(
            f"{split}: n={len(rows)}, bad_json={bad_json}, "
            f"missing_required_fields={missing}"
        )

    print("\nFirst train sample (decoded as UTF-8):")
    first = loaded["train"][0]
    print("title:", first["title"])
    print("author:", first["author"])
    print("prompt:", first["prompt"])
    print("completion:")
    print(first["completion"])

    print("\nExact-duplicate audit:")

    def ids(rows):
        return {x["id"] for x in rows if x.get("id")}

    def poems(rows):
        return {x["completion"].strip() for x in rows if x.get("completion")}

    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        id_overlap = ids(loaded[a]) & ids(loaded[b])
        poem_overlap = poems(loaded[a]) & poems(loaded[b])
        print(
            f"{a} vs {b}: id_overlap={len(id_overlap)}, "
            f"poem_text_overlap={len(poem_overlap)}"
        )

    all_rows = sum((loaded[s] for s in SPLITS), [])
    unique_ids = len(ids(all_rows))
    unique_poems = len(poems(all_rows))
    print(
        f"\nall rows={len(all_rows)}, unique non-empty ids={unique_ids}, "
        f"unique poem texts={unique_poems}"
    )


if __name__ == "__main__":
    main()
