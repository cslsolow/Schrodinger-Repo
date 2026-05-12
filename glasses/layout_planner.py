import ast
import random
from pathlib import Path

from glasses.layout_policy import get_repo_policy, infer_repo_name, is_protected_path, prefers_source_path
from glasses.trajectory_hotspots import summarize_hotspots


def _dir_depth(path: str) -> int:
    return len(Path(path).parts)


SEMANTIC_TARGET_POOLS = {
    "validator": ["screening", "policy", "constraints", "checksuite", "gatekeeping"],
    "validation": ["screening", "policy", "constraints", "checksuite", "gatekeeping"],
    "field": ["registry", "records", "catalog", "entries", "structures"],
    "form": ["intake", "composition", "entries", "submissions", "assembly"],
    "widget": ["presentation", "rendering", "composition", "assembly", "display"],
    "template": ["rendering", "composition", "presentation", "layout", "markup"],
    "html": ["rendering", "markup", "presentation", "composition", "layout"],
    "query": ["records", "relations", "ledger", "indexing", "selection"],
    "model": ["records", "relations", "catalog", "structures", "entities"],
    "auth": ["identity", "credentials", "access", "sessioning", "attestation"],
}

DEFAULT_SEMANTIC_TARGETS = [
    "analysis",
    "assembly",
    "catalog",
    "coordination",
    "dispatch",
    "engine",
    "inspection",
    "processing",
    "registry",
    "screening",
]


def resolve_instance_repo_root(repos_dir: Path, instance_id: str) -> Path:
    direct = Path(repos_dir) / f"swe-bench_{instance_id}"
    if direct.exists():
        return direct
    matches = sorted(Path(repos_dir).glob(f"*{instance_id}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No repo root found for {instance_id} under {repos_dir}")


def _iter_python_dirs(repo_root: Path) -> list[str]:
    dirs = set()
    for py_file in repo_root.rglob("*.py"):
        rel = py_file.relative_to(repo_root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        dirs.add(str(rel.parent))
    return sorted(dirs)


def _same_package_relative_dependencies(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()

    deps = set()
    parent = path.parent
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 1:
            continue

        if node.module:
            target = parent / f"{node.module}.py"
            if target.exists():
                deps.add(str(target))
            continue

        for alias in node.names:
            target = parent / f"{alias.name}.py"
            if target.exists():
                deps.add(str(target))
    return deps


def _is_anchor_module(path: Path) -> bool:
    text = path.read_text()
    anchor_markers = (
        "__file__",
        "Path(__file__)",
        "pathlib.Path(__file__)",
    )
    return any(marker in text for marker in anchor_markers)


def _expand_same_package_relative_closure(repo_root: Path, primary_dir: str, support_files: list[dict]) -> list[dict]:
    support_by_path = {item["path"]: dict(item) for item in support_files}
    pending = [Path(repo_root / item["path"]) for item in support_files]

    while pending:
        current = pending.pop()
        for dep in _same_package_relative_dependencies(current):
            rel = str(Path(dep).relative_to(repo_root))
            if str(Path(rel).parent) != primary_dir:
                continue
            if rel in support_by_path:
                continue
            support_by_path[rel] = {"path": rel, "score": 0.0, "evidence": {"closure": 1}}
            pending.append(Path(dep))

    return [support_by_path[path] for path in sorted(support_by_path)]


def _filter_anchor_modules(repo_root: Path, support_files: list[dict], primary_file: str) -> tuple[list[dict], list[dict]]:
    kept = []
    skipped = []
    for item in support_files:
        path = item["path"]
        full_path = repo_root / path
        if path != primary_file and _is_anchor_module(full_path):
            skipped.append({"path": path, "reason": "anchor_module"})
            continue
        kept.append(item)
    return kept, skipped


def _build_resource_map(repo_root: Path, primary_dir: str, target_dir: str, support_files: list[dict]) -> dict[str, str]:
    resource_map = {}
    source_root = Path(primary_dir)
    target_root = Path(target_dir)
    for item in support_files:
        source_path = Path(item["path"])
        module_stem = source_path.stem
        template_dir = repo_root / source_root / "templates" / "django" / "forms" / module_stem
        if not template_dir.exists():
            continue
        for resource in sorted(template_dir.rglob("*")):
            if resource.is_dir():
                continue
            rel = resource.relative_to(repo_root)
            mapped = target_root / "templates" / "django" / "forms" / module_stem / resource.name
            resource_map[str(rel)] = str(mapped)
    return resource_map


def _edit_distance_leq_one(left: str, right: str) -> bool:
    left = left.lower()
    right = right.lower()
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(left) == len(right):
            i += 1
            j += 1
        else:
            j += 1
    edits += (len(left) - i) + (len(right) - j)
    return edits <= 1


def _tokenize_semantic_hints(path: str) -> list[str]:
    tokens = []
    for part in Path(path).with_suffix("").parts:
        tokens.extend(piece for piece in part.replace("-", "_").split("_") if piece)
    return tokens


def _semantic_dir_names(primary_file: str, primary_dir: str, repo_name: str, repo_root: Path, seed: int) -> list[str]:
    tokens = _tokenize_semantic_hints(primary_file) + _tokenize_semantic_hints(primary_dir)
    names = []
    seen = set()
    for token in tokens:
        for candidate in SEMANTIC_TARGET_POOLS.get(token, []):
            if candidate not in seen:
                names.append(candidate)
                seen.add(candidate)
    for candidate in DEFAULT_SEMANTIC_TARGETS:
        if candidate not in seen:
            names.append(candidate)
            seen.add(candidate)

    existing_top_level = {p.name.lower() for p in repo_root.iterdir() if p.is_dir()}
    filtered = []
    for candidate in names:
        if candidate.lower() in existing_top_level:
            continue
        if _edit_distance_leq_one(candidate, repo_name):
            continue
        if any(_edit_distance_leq_one(candidate, token) for token in tokens):
            continue
        filtered.append(candidate)

    rng = random.Random(f"{seed}:{repo_name}:{primary_file}:{primary_dir}")
    rng.shuffle(filtered)
    return filtered


def _build_target_root(source_dir: str, target_dir: str) -> str:
    return target_dir


def _build_candidate_path_map(source_dir: str, target_dir: str, support_files: list[dict]) -> dict[str, str]:
    target_root = _build_target_root(source_dir, target_dir)
    path_map = {}
    for item in support_files:
        source_path = Path(item["path"])
        path_map[str(source_path)] = str(Path(target_root) / source_path.name)
    return path_map


def _path_map_conflicts(path_map: dict[str, str], repo_root: Path) -> bool:
    for source, target in path_map.items():
        if source == target:
            return True
        if (repo_root / target).exists():
            return True
    return False


def _build_module_map(path_map: dict[str, str]) -> dict[str, str]:
    return {
        ".".join(Path(source).with_suffix("").parts): ".".join(Path(target).with_suffix("").parts)
        for source, target in path_map.items()
    }


def _choose_target_dirs(source_dir: str, primary_file: str, repo_root: Path, repo_name: str, *, seed: int) -> list[dict]:
    policy = get_repo_policy(repo_name)
    candidates = []
    namespace_root = policy.namespace_root or Path(source_dir).parts[0]

    for name in _semantic_dir_names(primary_file, source_dir, repo_name, repo_root, seed):
        candidate = str(Path(namespace_root) / name)
        if candidate == source_dir:
            continue
        if is_protected_path(candidate + "/", policy):
            continue
        if (repo_root / candidate).exists():
            continue
        candidates.append({"path": candidate, "score": 10 - len(candidates)})

    candidates.sort(key=lambda item: (-item["score"], item["path"]))
    return candidates[: policy.max_target_dirs]


def build_layout_plan(traj_path: Path, repo_root: Path, *, top_k: int = 8, seed: int = 42) -> dict:
    hotspot_summary = summarize_hotspots(traj_path)
    repo_name = infer_repo_name(hotspot_summary["instance_id"])
    policy = get_repo_policy(repo_name)

    candidate_files = []
    skipped_files = []
    for item in hotspot_summary["files"]:
        path = item["path"]
        if is_protected_path(path, policy):
            skipped_files.append({"path": path, "reason": "protected"})
            continue
        if not prefers_source_path(path, policy):
            skipped_files.append({"path": path, "reason": "outside_preferred_source_prefixes"})
            continue
        candidate_files.append(item)

    candidate_files = candidate_files[:top_k]
    if not candidate_files:
        return {
            "instance_id": hotspot_summary["instance_id"],
            "repo_name": repo_name,
            "policy": policy.repo_name,
            "status": "no_candidate_files",
            "skipped_files": skipped_files,
        }

    primary_file = candidate_files[0]
    primary_dir = str(Path(primary_file["path"]).parent)
    support_files = []
    for item in candidate_files:
        if str(Path(item["path"]).parent) != primary_dir:
            continue
        if item["score"] < policy.min_support_score and item["path"] != primary_file["path"]:
            continue
        support_files.append(item)
    support_files = _expand_same_package_relative_closure(repo_root, primary_dir, support_files)
    support_files, anchor_skips = _filter_anchor_modules(repo_root, support_files, primary_file["path"])
    support_files = support_files[: policy.max_source_files]

    target_candidates = _choose_target_dirs(primary_dir, primary_file["path"], repo_root, repo_name, seed=seed)
    status = "ok" if target_candidates else "no_target_dirs"

    return {
        "instance_id": hotspot_summary["instance_id"],
        "repo_name": repo_name,
        "policy": policy.repo_name,
        "status": status,
        "unit_kind": "single_module" if len(support_files) == 1 else "module_group",
        "primary_file": primary_file,
        "primary_dir": primary_dir,
        "support_files": support_files,
        "target_candidates": target_candidates,
        "skipped_files": skipped_files + anchor_skips,
        "seed": seed,
    }


def materialize_layout_map(plan: dict, repo_root: Path) -> dict:
    if plan.get("status") != "ok":
        return {
            "instance_id": plan.get("instance_id"),
            "repo_name": plan.get("repo_name"),
            "status": "no_viable_target",
            "path_map": {},
            "reverse_map": {},
            "skipped_targets": [],
        }

    primary_dir = plan["primary_dir"]
    support_files = plan.get("support_files", [])
    skipped_targets = []
    for candidate in plan.get("target_candidates", []):
        target_dir = candidate["path"]
        target_root = _build_target_root(primary_dir, target_dir)
        path_map = _build_candidate_path_map(primary_dir, target_dir, support_files)
        if _path_map_conflicts(path_map, repo_root):
            skipped_targets.append({"path": target_dir, "reason": "target_conflict"})
            continue
        module_map = _build_module_map(path_map)
        return {
            "instance_id": plan.get("instance_id"),
            "repo_name": plan.get("repo_name"),
            "status": "ok",
            "path_based_mode": True,
            "source_dir": primary_dir,
            "target_dir": target_dir,
            "target_root": target_root,
            "path_map": path_map,
            "module_map": module_map,
            "resource_map": _build_resource_map(repo_root, primary_dir, target_dir, support_files),
            "reverse_map": {target: source for source, target in path_map.items()},
            "skipped_targets": skipped_targets,
        }

    return {
        "instance_id": plan.get("instance_id"),
        "repo_name": plan.get("repo_name"),
        "status": "no_viable_target",
        "source_dir": primary_dir,
        "path_map": {},
        "reverse_map": {},
        "skipped_targets": skipped_targets,
    }
