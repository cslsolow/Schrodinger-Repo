import builtins
import json
import re
from pathlib import Path


def _is_word_like(term):
    return re.fullmatch(r"[A-Za-z0-9_]+", term) is not None


def _iter_strings(value):
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
        return

    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)


def _contains_repo_term(value, repo_terms):
    texts = list(_iter_strings(value))
    if not texts:
        return False

    for term in repo_terms:
        if not term:
            continue

        if _is_word_like(term):
            pattern = re.compile(rf"\b{re.escape(term)}\b")
            if any(pattern.search(text) for text in texts):
                return True
            continue

        if any(term in text for text in texts):
            return True

    return False


def _message_text(message):
    return {
        "content": message.get("content"),
        "output": message.get("output"),
    }


def _load_messages(traj_path):
    data = json.loads(traj_path.read_text())
    if isinstance(data, dict):
        return data.get("messages", [])
    if isinstance(data, list):
        return data
    return []    


DEFAULT_FORBIDDEN_SHELL_TOKENS = {
    "awk",
    "cat",
    "cd",
    "cp",
    "diff",
    "echo",
    "find",
    "git",
    "grep",
    "head",
    "ls",
    "mkdir",
    "mv",
    "patch",
    "pip",
    "pip3",
    "pwd",
    "python",
    "python3",
    "rm",
    "sed",
    "sort",
    "tail",
    "wc",
}
DEFAULT_THIRD_PARTY_TOKENS = {
    "django",
    "fastapi",
    "flask",
    "numpy",
    "pandas",
    "pytest",
    "requests",
    "sklearn",
    "sqlalchemy",
    "tensorflow",
    "torch",
}
DEFAULT_BUILTIN_TOKENS = {name.lower() for name in dir(builtins)}


def summarize_leaks(run_dir: Path, repo_terms: list[str]) -> dict[str, int]:
    traj_paths = sorted(Path(run_dir).rglob("*.traj.json"))
    summary = {
        "traj_count": len(traj_paths),
        "user_message_leak_files": 0,
        "assistant_message_leak_files": 0,
        "command_leak_files": 0,
    }

    for traj_path in traj_paths:
        user_leak = False
        assistant_leak = False
        command_leak = False

        for message in _load_messages(traj_path):
            role = message.get("role")
            text_payload = _message_text(message)

            if role == "user" and _contains_repo_term(text_payload, repo_terms):
                user_leak = True

            if role == "assistant" and _contains_repo_term(text_payload, repo_terms):
                assistant_leak = True

            for action in message.get("extra", {}).get("actions", []):
                if _contains_repo_term(action.get("command", ""), repo_terms):
                    command_leak = True

        summary["user_message_leak_files"] += int(user_leak)
        summary["assistant_message_leak_files"] += int(assistant_leak)
        summary["command_leak_files"] += int(command_leak)

    return summary


def _find_repo_instance_dir(repo_maps_dir: Path, repo_name: str) -> Path | None:
    candidates = sorted(
        p for p in Path(repo_maps_dir).iterdir() if p.is_dir() and p.name.startswith(f"{repo_name}__") and (p / "token_mapping.json").exists()
    )
    return candidates[0] if candidates else None


def summarize_bundle_quality(
    repo_maps_dir: Path,
    repo_name: str,
    *,
    variant_name: str | None = None,
    forbidden_shell_tokens: set[str] | None = None,
    forbidden_third_party_tokens: set[str] | None = None,
    forbidden_builtin_tokens: set[str] | None = None,
) -> dict:
    repo_maps_dir = Path(repo_maps_dir)
    bundle_root = repo_maps_dir / repo_name
    bundle_dir = bundle_root
    index_path = bundle_root / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        variants = index.get("variants", [])
        selected = None
        if variant_name is not None:
            selected = next((item for item in variants if item.get("name") == variant_name), None)
        if selected is None and variants:
            selected = variants[0]
        if selected is not None:
            bundle_dir = bundle_root / selected["name"]
    instance_dir = _find_repo_instance_dir(repo_maps_dir, repo_name)

    token_mapping = {}
    if instance_dir:
        token_mapping = json.loads((instance_dir / "token_mapping.json").read_text())

    namespace_symbol_map = json.loads((bundle_dir / "namespace_symbol_map.json").read_text())
    namespace_path_map = json.loads((bundle_dir / "namespace_path_map.json").read_text())
    identity_map = json.loads((bundle_dir / "identity_map.json").read_text())

    forbidden_shell_tokens = forbidden_shell_tokens or DEFAULT_FORBIDDEN_SHELL_TOKENS
    forbidden_third_party_tokens = forbidden_third_party_tokens or DEFAULT_THIRD_PARTY_TOKENS
    forbidden_builtin_tokens = forbidden_builtin_tokens or DEFAULT_BUILTIN_TOKENS

    def changed_items(tokens: set[str]) -> list[dict[str, str]]:
        return [
            {"real": token, "virtual": token_mapping[token]}
            for token in sorted(tokens)
            if token in token_mapping and token_mapping[token] != token
        ]

    return {
        "repo_name": repo_name,
        "bundle_dir": str(bundle_dir),
        "instance_dir": str(instance_dir) if instance_dir else None,
        "identity_total": len(identity_map),
        "identity_namespace_overlap": sorted(set(identity_map) & (set(namespace_symbol_map) | set(namespace_path_map))),
        "symbol_total": len(namespace_symbol_map),
        "symbol_changed": sum(1 for k, v in namespace_symbol_map.items() if k != v),
        "path_total": len(namespace_path_map),
        "path_changed": sum(1 for k, v in namespace_path_map.items() if k != v),
        "token_total": len(token_mapping),
        "token_changed": sum(1 for k, v in token_mapping.items() if k != v),
        "forbidden_shell_changed": changed_items(forbidden_shell_tokens),
        "forbidden_third_party_changed": changed_items(forbidden_third_party_tokens),
        "forbidden_builtin_changed": changed_items(forbidden_builtin_tokens),
    }


def validate_bundle_quality(repo_maps_dir: Path, repo_name: str, *, variant_name: str | None = None) -> dict:
    summary = summarize_bundle_quality(repo_maps_dir, repo_name, variant_name=variant_name)
    violations = []
    if summary["identity_namespace_overlap"]:
        violations.append(f"identity/namespace overlap: {summary['identity_namespace_overlap']}")
    if summary["forbidden_shell_changed"]:
        violations.append(f"shell tokens mapped: {summary['forbidden_shell_changed']}")
    if summary["forbidden_third_party_changed"]:
        violations.append(f"third-party tokens mapped: {summary['forbidden_third_party_changed']}")
    if summary["forbidden_builtin_changed"]:
        violations.append(f"builtin tokens mapped: {summary['forbidden_builtin_changed']}")

    if violations:
        raise ValueError("; ".join(violations))
    return summary
