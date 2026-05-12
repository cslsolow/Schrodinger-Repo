import json
import random
from pathlib import Path


def _merge_layer(forward: dict[str, str], layer: dict[str, str]) -> None:
    duplicate_keys = set(forward) & set(layer)
    if duplicate_keys:
        duplicates = ", ".join(sorted(duplicate_keys))
        raise ValueError(f"Duplicate mapping keys across enabled layers: {duplicates}")
    forward.update(layer)


def load_mapping_bundle(bundle_dir: Path) -> dict:
    return {
        "identity_map": json.loads((bundle_dir / "identity_map.json").read_text()),
        "namespace_symbol_map": json.loads((bundle_dir / "namespace_symbol_map.json").read_text()),
        "namespace_path_map": json.loads((bundle_dir / "namespace_path_map.json").read_text()),
        "mapping_meta": json.loads((bundle_dir / "mapping_meta.json").read_text()),
    }


def load_mapping_index(bundle_root: Path) -> dict | None:
    index_path = bundle_root / "index.json"
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text())


def select_mapping_bundle_dir(bundle_root: Path, semantic_seed: int) -> tuple[Path, dict | None]:
    index = load_mapping_index(bundle_root)
    if not index:
        return bundle_root, None

    variants = index.get("variants", [])
    if not variants:
        return bundle_root, None

    repo_name = index.get("repo_name") or bundle_root.name
    variant_index = random.Random(f"{repo_name}:{semantic_seed}:l2").randrange(len(variants))
    selected_variant = variants[variant_index]
    return bundle_root / selected_variant["name"], selected_variant


def compose_forward_map(bundle: dict, enabled_layers: list[str]) -> dict[str, str]:
    forward: dict[str, str] = {}
    if "identity_l1" in enabled_layers:
        _merge_layer(forward, bundle["identity_map"])
    if "namespace_l2" in enabled_layers:
        _merge_layer(forward, bundle["namespace_symbol_map"])
        _merge_layer(forward, bundle["namespace_path_map"])
    return forward
