from pathlib import Path
import re


def _module_name(path: str) -> str:
    p = Path(path)
    return ".".join(p.with_suffix("").parts)


def _parse_imported_items(imported: str) -> list[tuple[str, str]]:
    items = []
    for raw_item in imported.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        base_name = raw_item.split(" as ", 1)[0].strip()
        items.append((raw_item, base_name))
    return items


def _rewrite_import_line(line: str, module_map: dict[str, str]) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    if stripped.startswith("import "):
        updated = stripped
        for source_module, target_module in sorted(module_map.items(), key=lambda item: -len(item[0])):
            updated = updated.replace(source_module, target_module)
        return indent + updated

    if stripped.startswith("from "):
        match = re.match(r"from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)", stripped)
        if not match:
            return line
        from_module, imported = match.groups()
        imported_items = _parse_imported_items(imported)
        updated = stripped
        for source_module, target_module in sorted(module_map.items(), key=lambda item: -len(item[0])):
            source_parent, _, source_leaf = source_module.rpartition(".")
            target_parent, _, target_leaf = target_module.rpartition(".")
            if from_module == source_module:
                updated = updated.replace(source_module, target_module, 1)
                break
            if from_module != source_parent:
                continue

            moved_by_target_parent = {}
            kept_items = []
            for raw_item, base_name in imported_items:
                if base_name != source_leaf:
                    kept_items.append(raw_item)
                    continue
                replacement = raw_item
                if target_leaf != source_leaf:
                    alias = raw_item.split(" as ", 1)[1].strip() if " as " in raw_item else source_leaf
                    replacement = f"{target_leaf} as {alias}"
                moved_by_target_parent.setdefault(target_parent, []).append(replacement)

            if not moved_by_target_parent:
                continue

            lines = []
            if kept_items:
                lines.append(f"{indent}from {source_parent} import {', '.join(kept_items)}\n")
            for parent, items in moved_by_target_parent.items():
                lines.append(f"{indent}from {parent} import {', '.join(items)}\n")
            return "".join(lines)
        return indent + updated

    return line


def _rewrite_python_imports(repo_root: Path, module_map: dict[str, str]) -> None:
    for py_file in repo_root.rglob("*.py"):
        text = py_file.read_text()
        updated = "".join(_rewrite_import_line(line, module_map) for line in text.splitlines(keepends=True))
        if updated != text:
            py_file.write_text(updated)


def apply_layout_map(repo_root: Path, layout_map: dict) -> dict:
    if layout_map.get("status") != "ok":
        return {"status": "skipped"}

    repo_root = Path(repo_root)
    path_map = layout_map.get("path_map", {})
    module_map = layout_map.get("module_map") or {_module_name(source): _module_name(target) for source, target in path_map.items()}
    resource_map = layout_map.get("resource_map", {}) if layout_map.get("move_resources") else {}

    if not layout_map.get("path_based_mode"):
        _rewrite_python_imports(repo_root, module_map)

    target_root = layout_map.get("target_root")
    if target_root:
        init_path = repo_root / target_root / "__init__.py"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.touch(exist_ok=True)

    for source, target in path_map.items():
        source_path = repo_root / source
        target_path = repo_root / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(target_path)

    for source, target in resource_map.items():
        source_path = repo_root / source
        target_path = repo_root / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(target_path)

    return {
        "status": "ok",
        "moved_files": len(path_map),
        "moved_resources": len(resource_map),
        "path_map": path_map,
    }
