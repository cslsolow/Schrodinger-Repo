#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--shard-root", action="append", dest="shard_roots", required=True)
    return parser.parse_args()


def load_index(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_index(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root)
    base_root.mkdir(parents=True, exist_ok=True)
    base_index_path = base_root / "index.json"
    base_index = load_index(base_index_path)

    merged = 0
    for shard_root_str in args.shard_roots:
        shard_root = Path(shard_root_str)
        shard_index = load_index(shard_root / "index.json")
        for instance_id, status in shard_index.items():
            src_dir = shard_root / instance_id
            dst_dir = base_root / instance_id
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            base_index[instance_id] = status
            merged += 1

    write_index(base_index_path, base_index)
    print("merged_entries =", merged)
    print("base_index_count =", len(base_index))


if __name__ == "__main__":
    main()
