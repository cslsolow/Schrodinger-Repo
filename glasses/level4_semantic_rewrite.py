#!/usr/bin/env python3

import re
from pathlib import Path


def extract_patch_target_ranges(patch_text: str) -> dict[str, list[tuple[int, int]]]:
    ranges = {}
    current_path = None
    for line in patch_text.splitlines():
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if match:
            old_path, new_path = match.groups()
            path = new_path if new_path != "/dev/null" else old_path
            current_path = path if path.endswith(".py") and not is_protected_path(path) else None
            continue
        if current_path is None:
            continue
        hunk = re.match(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
        if hunk:
            start = int(hunk.group(1))
            count = int(hunk.group(2) or "1")
            end = start + max(count, 1) - 1
            ranges.setdefault(current_path, []).append((start, end))
    return ranges


def is_protected_path(path: str) -> bool:
    parts = Path(path).parts
    name = parts[-1] if parts else path
    if name in {"__init__.py", "conftest.py", "setup.py"}:
        return True
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    blocked = {"tests", "test", "migrations"}
    if any(part in blocked for part in parts):
        return True
    for left, right in zip(parts, parts[1:]):
        if left == "management" and right == "commands":
            return True
    risky_names = {"apps.py", "registry.py", "loading.py", "signals.py"}
    return name in risky_names
