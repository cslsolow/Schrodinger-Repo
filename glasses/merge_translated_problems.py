#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Error: file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"Error: expected JSON object at root: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge translated problems JSON files.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_json(args.base)
    extra = load_json(args.extra)

    merged = dict(base)
    merged.update(extra)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"merged {len(base)} + {len(extra)} -> {len(merged)} entries: {args.output}")


if __name__ == "__main__":
    main()
