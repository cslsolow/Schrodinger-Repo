import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PROTECTED_PREFIXES = (
    "tests/",
    "*/migrations/",
    "migrations/",
    "templates/",
    "static/",
    "locale/",
    "management/commands/",
)

DEFAULT_SCORE_WEIGHTS = {
    "edited": 5.0,
    "read": 3.0,
    "stack": 2.0,
    "search": 1.0,
}

_PATCH_FILE_RE = re.compile(r"^\+\+\+ b/([^\s]+)", re.MULTILINE)
_TRACEBACK_FILE_RE = re.compile(r'File "(?:/testbed/)?([^"\n]+)"')
_PATH_RE = re.compile(r"(?<![\w./-])(?:/testbed/)?((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py)\b")


def _load_traj(traj_path: Path) -> dict:
    data = json.loads(Path(traj_path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict trajectory, got {type(data).__name__}")
    return data


def _iter_messages(traj: dict) -> list[dict]:
    messages = traj.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _normalize_repo_path(path: str) -> str | None:
    path = path.strip().strip("'\"")
    if not path:
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    if path.startswith("/testbed/"):
        path = path[len("/testbed/") :]
    elif path.startswith("testbed/"):
        path = path[len("testbed/") :]
    path = path.lstrip("./")
    if path.startswith("../") or path.startswith("/"):
        return None
    if not path.endswith(".py"):
        return None
    return path


def _iter_texts(value):
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_texts(item)
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_texts(nested)


def _extract_paths_from_text(text: str) -> list[str]:
    paths = []
    for match in _PATH_RE.findall(text):
        normalized = _normalize_repo_path(match)
        if normalized:
            paths.append(normalized)
    return paths


def _protected_path(path: str, protected_prefixes: tuple[str, ...]) -> bool:
    for prefix in protected_prefixes:
        if prefix.startswith("*/"):
            needle = prefix[2:]
            if f"/{needle}" in f"/{path}":
                return True
            continue
        if path.startswith(prefix):
            return True
    return False


def _message_actions(message: dict) -> list[dict]:
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return []
    actions = extra.get("actions", [])
    if not isinstance(actions, list):
        return []
    return [a for a in actions if isinstance(a, dict)]


def extract_hotspot_evidence(
    traj_path: Path,
    *,
    protected_prefixes: tuple[str, ...] = DEFAULT_PROTECTED_PREFIXES,
) -> dict[str, list[str]]:
    traj = _load_traj(traj_path)
    evidence = defaultdict(list)

    for message in _iter_messages(traj):
        role = message.get("role")

        if role == "exit":
            content = message.get("content", "")
            if isinstance(content, str):
                for path in _PATCH_FILE_RE.findall(content):
                    normalized = _normalize_repo_path(path)
                    if normalized and not _protected_path(normalized, protected_prefixes):
                        evidence["edited"].append(normalized)

        if role == "tool":
            for text in _iter_texts(message.get("content")):
                stack_paths = set()
                for path in _TRACEBACK_FILE_RE.findall(text):
                    normalized = _normalize_repo_path(path)
                    if normalized and not _protected_path(normalized, protected_prefixes):
                        evidence["stack"].append(normalized)
                        stack_paths.add(normalized)
                for path in _extract_paths_from_text(text):
                    if _protected_path(path, protected_prefixes):
                        continue
                    if path in stack_paths:
                        continue
                    if ":" in text and path in text:
                        evidence["search"].append(path)
                    else:
                        evidence["stack"].append(path)

        for action in _message_actions(message):
            command = action.get("command", "")
            if not isinstance(command, str):
                continue
            for path in _extract_paths_from_text(command):
                if not _protected_path(path, protected_prefixes):
                    evidence["read"].append(path)

    return {kind: sorted(paths) for kind, paths in evidence.items()}


def summarize_hotspots(
    traj_path: Path,
    *,
    weights: dict[str, float] | None = None,
    protected_prefixes: tuple[str, ...] = DEFAULT_PROTECTED_PREFIXES,
    top_k: int | None = None,
) -> dict:
    weights = weights or DEFAULT_SCORE_WEIGHTS
    evidence = extract_hotspot_evidence(traj_path, protected_prefixes=protected_prefixes)
    counts_by_kind = {kind: Counter(paths) for kind, paths in evidence.items()}
    all_paths = sorted({path for paths in evidence.values() for path in paths})

    files = []
    for path in all_paths:
        score = 0.0
        kinds = {}
        for kind, counter in counts_by_kind.items():
            count = counter.get(path, 0)
            if not count:
                continue
            kinds[kind] = count
            score += weights.get(kind, 0.0)
            if count > 1:
                score += 0.5 * (count - 1)
        files.append({"path": path, "score": score, "evidence": kinds})

    files.sort(key=lambda item: (-item["score"], item["path"]))
    if top_k is not None:
        files = files[:top_k]

    directories = Counter()
    for item in files:
        directories[str(Path(item["path"]).parent)] += item["score"]

    return {
        "instance_id": Path(traj_path).stem.replace(".traj", ""),
        "traj_path": str(traj_path),
        "files": files,
        "directories": [
            {"path": path, "score": score}
            for path, score in sorted(directories.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
