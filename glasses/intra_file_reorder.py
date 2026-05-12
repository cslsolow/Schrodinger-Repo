import ast
import json
import random
from pathlib import Path

from glasses.layout_policy import get_repo_policy, infer_repo_name, is_protected_path, prefers_source_path
from glasses.trajectory_hotspots import summarize_hotspots


def _node_start_lineno(node) -> int:
    starts = [node.lineno]
    for decorator in getattr(node, "decorator_list", []):
        starts.append(decorator.lineno)
    return min(starts)


def _is_test_path(path: str) -> bool:
    p = Path(path)
    if any(part == "tests" for part in p.parts):
        return True
    name = p.name
    stem = p.stem
    return name.startswith("test_") or stem.endswith("_test")


def _is_top_level_reorderable(node) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _is_method_reorderable(node) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _group_reorderable_runs(body, predicate) -> list[list]:
    runs = []
    current = []
    previous_index = None
    for index, node in enumerate(body):
        if not predicate(node):
            if current:
                runs.append(current)
                current = []
            previous_index = None
            continue
        if current and previous_index is not None and index != previous_index + 1:
            runs.append(current)
            current = []
        current.append(node)
        previous_index = index
    if current:
        runs.append(current)
    return runs


def _class_key(prefix: str, node) -> str:
    key = node.name
    return f"{prefix}::{key}" if prefix else key


def _collect_class_run_lengths(body, prefix: str = "") -> dict[str, list[int]]:
    result = {}
    for node in body:
        if not isinstance(node, ast.ClassDef):
            continue
        key = _class_key(prefix, node)
        result[key] = [len(run) for run in _group_reorderable_runs(node.body, _is_method_reorderable)]
        result.update(_collect_class_run_lengths(node.body, key))
    return result


def _loaded_names(node) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    return names


def _function_definition_names(node) -> set[str]:
    names = set()
    for decorator in getattr(node, "decorator_list", []):
        names.update(_loaded_names(decorator))
    if getattr(node, "returns", None) is not None:
        names.update(_loaded_names(node.returns))
    if getattr(node, "type_params", None):
        for param in node.type_params:
            names.update(_loaded_names(param))

    args = getattr(node, "args", None)
    if args is None:
        return names

    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg and args.vararg.annotation is not None:
        names.update(_loaded_names(args.vararg.annotation))
    if args.kwarg and args.kwarg.annotation is not None:
        names.update(_loaded_names(args.kwarg.annotation))

    for arg in all_args:
        if arg.annotation is not None:
            names.update(_loaded_names(arg.annotation))
    for default in list(args.defaults) + [item for item in args.kw_defaults if item is not None]:
        names.update(_loaded_names(default))
    return names


def _class_definition_names(node) -> set[str]:
    names = set()
    for decorator in getattr(node, "decorator_list", []):
        names.update(_loaded_names(decorator))
    for base in getattr(node, "bases", []):
        names.update(_loaded_names(base))
    for keyword in getattr(node, "keywords", []):
        if keyword.value is not None:
            names.update(_loaded_names(keyword.value))

    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.update(_function_definition_names(stmt))
            continue
        if isinstance(stmt, ast.ClassDef):
            names.update(_class_definition_names(stmt))
            continue
        names.update(_loaded_names(stmt))
    return names


def _definition_time_names(node) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _function_definition_names(node)
    if isinstance(node, ast.ClassDef):
        return _class_definition_names(node)
    return set()


def _build_run_specs(body, predicate) -> list[dict]:
    specs = []
    for run in _group_reorderable_runs(body, predicate):
        positions = {}
        for index, node in enumerate(run):
            positions.setdefault(node.name, []).append(index)
        deps = []
        for index, node in enumerate(run):
            required = set()
            for name in _definition_time_names(node):
                candidates = [candidate for candidate in positions.get(name, []) if candidate < index]
                if not candidates:
                    continue
                required.add(candidates[-1])
            deps.append(sorted(required))
        specs.append({"length": len(run), "deps": deps})
    return specs


def _collect_class_run_specs(body, prefix: str = "") -> dict[str, list[dict]]:
    result = {}
    for node in body:
        if not isinstance(node, ast.ClassDef):
            continue
        key = _class_key(prefix, node)
        result[key] = _build_run_specs(node.body, _is_method_reorderable)
        result.update(_collect_class_run_specs(node.body, key))
    return result


def analyze_reorderable_file(path: Path) -> dict:
    source = Path(path).read_text()
    tree = ast.parse(source)
    top_run_specs = _build_run_specs(tree.body, _is_top_level_reorderable)
    class_run_specs = _collect_class_run_specs(tree.body)
    movable_units = sum(max(spec["length"] - 1, 0) for spec in top_run_specs)
    movable_units += sum(max(spec["length"] - 1, 0) for runs in class_run_specs.values() for spec in runs)
    return {
        "top_run_specs": top_run_specs,
        "class_run_specs": class_run_specs,
        "movable_units": movable_units,
    }


def _random_topological_order(run_spec: dict, rng: random.Random) -> list[int]:
    length = run_spec["length"]
    deps = [set(items) for items in run_spec["deps"]]
    dependents = {index: set() for index in range(length)}
    indegree = [0] * length
    for index, required in enumerate(deps):
        indegree[index] = len(required)
        for parent in required:
            dependents[parent].add(index)

    available = [index for index in range(length) if indegree[index] == 0]
    order = []
    while available:
        choice = rng.choice(sorted(available))
        available.remove(choice)
        order.append(choice)
        for child in sorted(dependents[choice]):
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(child)

    if len(order) != length:
        return list(range(length))

    base = list(range(length))
    if length <= 1:
        return order
    for _ in range(32):
        candidate_rng = random.Random(f"{rng.random()}:{_}")
        candidate = []
        indegree = [len(items) for items in deps]
        available = [index for index in range(length) if indegree[index] == 0]
        while available:
            choice = candidate_rng.choice(sorted(available))
            available.remove(choice)
            candidate.append(choice)
            for child in sorted(dependents[choice]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    available.append(child)
        if candidate != base:
            return candidate
    return order


def _build_file_variant_orders(file_spec: dict, *, variant_seed: str) -> dict:
    top_runs = []
    for index, run_spec in enumerate(file_spec["top_run_specs"]):
        rng = random.Random(f"{variant_seed}:top:{index}")
        top_runs.append(_random_topological_order(run_spec, rng))

    class_runs = {}
    for class_key, run_specs in sorted(file_spec["class_run_specs"].items()):
        class_runs[class_key] = []
        for index, run_spec in enumerate(run_specs):
            rng = random.Random(f"{variant_seed}:class:{class_key}:{index}")
            class_runs[class_key].append(_random_topological_order(run_spec, rng))

    return {
        "top_runs": top_runs,
        "class_runs": class_runs,
    }


def _variant_changes_something(file_orders: dict) -> bool:
    for order in file_orders["top_runs"]:
        if order != list(range(len(order))):
            return True
    for runs in file_orders["class_runs"].values():
        for order in runs:
            if order != list(range(len(order))):
                return True
    return False


def _invert_orders(orders: list[list[int]]) -> list[list[int]]:
    inverted = []
    for order in orders:
        inverse = [0] * len(order)
        for new_index, old_index in enumerate(order):
            inverse[old_index] = new_index
        inverted.append(inverse)
    return inverted


def _render_scope(lines: list[str], body: list, *, scope_start: int, scope_end: int, predicate, run_orders: list[list[int]], class_orders: dict, class_prefix: str = "") -> str:
    runs = _group_reorderable_runs(body, predicate)
    output = []
    cursor = scope_start

    for run_index, run in enumerate(runs):
        run_start = _node_start_lineno(run[0])
        output.append("".join(lines[cursor - 1 : run_start - 1]))

        block_texts = []
        for index, node in enumerate(run):
            core_start = _node_start_lineno(node)
            core_end = node.end_lineno
            if isinstance(node, ast.ClassDef):
                core_text = _render_class_block(lines, node, class_orders, class_prefix)
            else:
                core_text = "".join(lines[core_start - 1 : core_end])
            next_start = _node_start_lineno(run[index + 1]) if index + 1 < len(run) else None
            trailing_end = next_start - 1 if next_start else core_end
            trailing_text = "".join(lines[core_end:trailing_end]) if trailing_end > core_end else ""
            block_texts.append(core_text + trailing_text)

        order = run_orders[run_index] if run_index < len(run_orders) else list(range(len(block_texts)))
        output.extend(block_texts[position] for position in order)
        cursor = run[-1].end_lineno + 1

    output.append("".join(lines[cursor - 1 : scope_end]))
    return "".join(output)


def _render_class_block(lines: list[str], class_node, class_orders: dict, prefix: str = "") -> str:
    key = _class_key(prefix, class_node)
    run_orders = class_orders.get(key, [])
    return _render_scope(
        lines,
        class_node.body,
        scope_start=_node_start_lineno(class_node),
        scope_end=class_node.end_lineno,
        predicate=_is_method_reorderable,
        run_orders=run_orders,
        class_orders=class_orders,
        class_prefix=key,
    )


def _render_file_variant(source: str, file_orders: dict) -> str:
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    return _render_scope(
        lines,
        tree.body,
        scope_start=1,
        scope_end=len(lines),
        predicate=_is_top_level_reorderable,
        run_orders=file_orders.get("top_runs", []),
        class_orders=file_orders.get("class_runs", {}),
    )


def invert_file_variant_orders(file_orders: dict) -> dict:
    return {
        "top_runs": _invert_orders(file_orders.get("top_runs", [])),
        "class_runs": {
            class_key: _invert_orders(orders)
            for class_key, orders in file_orders.get("class_runs", {}).items()
        },
    }


def restore_original_file_order(source: str, file_orders: dict) -> str:
    return _render_file_variant(source, invert_file_variant_orders(file_orders))


def build_intra_file_reorder_plan(traj_path: Path, repo_root: Path, *, top_k: int = 8, variant_count: int = 5, seed: int = 42) -> dict:
    hotspot_summary = summarize_hotspots(traj_path)
    repo_name = infer_repo_name(hotspot_summary["instance_id"])
    policy = get_repo_policy(repo_name)

    candidate_files = []
    skipped_files = []
    for item in hotspot_summary["files"]:
        path = item["path"]
        if not path.endswith(".py"):
            skipped_files.append({"path": path, "reason": "not_python"})
            continue
        if _is_test_path(path):
            skipped_files.append({"path": path, "reason": "test_file"})
            continue
        if is_protected_path(path, policy):
            skipped_files.append({"path": path, "reason": "protected"})
            continue
        if not prefers_source_path(path, policy):
            skipped_files.append({"path": path, "reason": "outside_preferred_source_prefixes"})
            continue
        full_path = Path(repo_root) / path
        if not full_path.exists():
            skipped_files.append({"path": path, "reason": "missing"})
            continue
        analysis = analyze_reorderable_file(full_path)
        if analysis["movable_units"] <= 0:
            skipped_files.append({"path": path, "reason": "no_reorderable_content"})
            continue
        candidate_files.append(
            {
                "path": path,
                "score": item["score"],
                "evidence": item["evidence"],
                "analysis": analysis,
            }
        )

    candidate_files = candidate_files[:top_k]
    if not candidate_files:
        return {
            "instance_id": hotspot_summary["instance_id"],
            "repo_name": repo_name,
            "policy": policy.repo_name,
            "status": "no_candidate_files",
            "files": [],
            "variants": [],
            "skipped_files": skipped_files,
            "seed": seed,
        }

    variants = []
    seen = set()
    nonce = 0
    while len(variants) < variant_count and nonce < 512:
        file_orders = {}
        for file_item in candidate_files:
            variant_seed = f"{seed}:{hotspot_summary['instance_id']}:{file_item['path']}:{nonce}"
            file_orders[file_item["path"]] = _build_file_variant_orders(file_item["analysis"], variant_seed=variant_seed)
        signature = json.dumps(file_orders, sort_keys=True)
        if signature not in seen and any(_variant_changes_something(orders) for orders in file_orders.values()):
            variants.append(
                {
                    "index": len(variants),
                    "nonce": nonce,
                    "file_orders": file_orders,
                }
            )
            seen.add(signature)
        nonce += 1

    status = "ok" if len(variants) == variant_count else "limited_variants"
    return {
        "instance_id": hotspot_summary["instance_id"],
        "repo_name": repo_name,
        "policy": policy.repo_name,
        "status": status,
        "files": candidate_files,
        "variants": variants,
        "skipped_files": skipped_files,
        "seed": seed,
    }


def select_intra_file_variant(plan: dict, runtime_seed: int) -> dict | None:
    variants = plan.get("variants", [])
    if not variants:
        return None
    index = random.Random(f"{plan.get('instance_id')}:{plan.get('seed', 42)}:{runtime_seed}").randrange(len(variants))
    return variants[index]


def materialize_intra_file_variant(repo_root: Path, plan: dict, *, runtime_seed: int) -> dict:
    variant = select_intra_file_variant(plan, runtime_seed)
    if not variant:
        return {"status": "skipped"}

    repo_root = Path(repo_root)
    files = {}
    rewritten_files = []
    for file_item in plan.get("files", []):
        path = file_item["path"]
        full_path = repo_root / path
        source = full_path.read_text()
        updated = _render_file_variant(source, variant["file_orders"][path])
        if updated != source:
            files[path] = updated
            rewritten_files.append(path)

    return {
        "status": "ok",
        "variant_index": variant["index"],
        "file_orders": variant["file_orders"],
        "files": files,
        "rewritten_files": rewritten_files,
    }


def apply_intra_file_reorder(repo_root: Path, plan: dict, *, runtime_seed: int) -> dict:
    materialized = materialize_intra_file_variant(repo_root, plan, runtime_seed=runtime_seed)
    if materialized.get("status") != "ok":
        return materialized

    repo_root = Path(repo_root)
    for path, updated in materialized["files"].items():
        (repo_root / path).write_text(updated)

    return {
        "status": "ok",
        "variant_index": materialized["variant_index"],
        "file_orders": materialized["file_orders"],
        "rewritten_files": materialized["rewritten_files"],
    }
