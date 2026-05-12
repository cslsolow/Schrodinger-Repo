#!/usr/bin/env python3

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import Dataset
from openai import OpenAI

from verified_dataset_paths import default_verified_arrow

SYSTEM_PARAPHRASE_ISSUE_EN = """\
You rewrite a GitHub / SWE-bench English issue for a robustness benchmark.

Requirements:
- Preserve the same technical meaning, bug, expected behavior, actual behavior, and constraints.
- Change wording and sentence structure substantially.
- Do not add facts or remove important technical details.
- Keep code identifiers, file paths, API names, class/function names, issue numbers, and fenced code blocks verbatim when needed.
- Remove person-identifying details in prose when they are not technically necessary.

Return only the rewritten English issue text.
"""


def paraphrase_issue_en_openai(text: str, model: str) -> str:
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PARAPHRASE_ISSUE_EN},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def load_verified_instances() -> list[dict]:
    arrow = default_verified_arrow()
    ds = Dataset.from_file(str(arrow))
    return list(ds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate SWE-bench verified problem statements by filter-file.")
    parser.add_argument("--filter-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Error: OPENAI_API_KEY is required")
    if args.workers < 1:
        raise SystemExit("Error: --workers must be >= 1")
    if not args.filter_file.exists():
        raise SystemExit(f"Error: filter file not found: {args.filter_file}")

    target_ids = [line.strip() for line in args.filter_file.read_text().splitlines() if line.strip()]
    target_set = set(target_ids)

    instances = load_verified_instances()
    selected = [inst for inst in instances if inst["instance_id"] in target_set]
    selected.sort(key=lambda x: target_ids.index(x["instance_id"]))

    results = {}
    if args.resume and args.output.exists():
        raw = args.output.read_text().strip()
        if raw:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                results = loaded

    lock = threading.Lock()
    finished = 0
    total = len(selected)

    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    def run_one(instance: dict) -> None:
        nonlocal finished
        instance_id = instance["instance_id"]
        if results.get(instance_id, {}).get("translated") is not None:
            with lock:
                finished += 1
                print(f"[{finished}/{total}] skip {instance_id}")
            return

        original = instance["problem_statement"]
        try:
            translated = paraphrase_issue_en_openai(original, args.model)
            record = {
                "original": original,
                "translated_zh": "",
                "translated": translated,
            }
            status = f"ok {instance_id}"
        except Exception as exc:
            record = {
                "original": original,
                "translated": None,
                "error": str(exc),
            }
            status = f"error {instance_id}: {exc}"

        with lock:
            results[instance_id] = record
            finished += 1
            print(f"[{finished}/{total}] {status}")
            if finished % 10 == 0:
                persist()

    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(selected)))) as executor:
        list(executor.map(run_one, selected))

    persist()
    success = sum(1 for v in results.values() if isinstance(v, dict) and v.get("translated") is not None)
    print(f"Done: {success}/{len(selected)} -> {args.output}")


if __name__ == "__main__":
    main()
