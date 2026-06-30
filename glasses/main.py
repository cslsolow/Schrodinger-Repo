import json
import argparse
from pathlib import Path
from check_observation_leaks import (
    DEFAULT_BUILTIN_TOKENS,
    DEFAULT_FORBIDDEN_SHELL_TOKENS,
    DEFAULT_THIRD_PARTY_TOKENS,
    validate_bundle_quality,
)

# Must match validate_bundle_quality / summarize_bundle_quality: never remap these tokens.
FORBIDDEN_IDENTITY_TOKENS = (
    DEFAULT_FORBIDDEN_SHELL_TOKENS
    | DEFAULT_THIRD_PARTY_TOKENS
    | DEFAULT_BUILTIN_TOKENS
)
from extractor import RepoIdentifierExtractor
from mapper import SemanticMapper
from rich.console import Console
from rich.table import Table

console = Console()
FIXED_REPOSITORY_ALIAS = "working_repository"

# Maps SWE-bench repo owner/name → importable Python package name.
# When the two differ, both need to map to working_repository.
REPO_PACKAGE_ALIASES: dict[str, str] = {
    "scikit-learn": "sklearn",
    "pytest-dev": "pytest",
    "mwaskom": "seaborn",
    "pallets": "flask",
    "sphinx-doc": "sphinx",
    "pylint-dev": "pylint",
    "pydata": "xarray",
    "psf": "requests",
}


def _infer_importable_repo_roots(repo_path: Path) -> list[str]:
    roots = set()
    ignored = {".git", "__pycache__", "migrations", "tests", "testing"}

    for child in repo_path.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in ignored:
            continue
        if (child / "__init__.py").exists():
            roots.add(child.name)

    src_dir = repo_path / "src"
    if src_dir.is_dir():
        for child in src_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(".") or child.name in ignored:
                continue
            if (child / "__init__.py").exists():
                roots.add(child.name)

    return sorted(roots)


def _resolve_repo_identity_names(project_repo: str, repo_path: Path) -> tuple[list[str], str]:
    inferred_roots = _infer_importable_repo_roots(repo_path)
    fallback_alias = REPO_PACKAGE_ALIASES.get(project_repo)

    identity_names = [project_repo]
    for root in inferred_roots:
        if root not in identity_names:
            identity_names.append(root)

    if len(inferred_roots) == 1:
        canonical = inferred_roots[0]
    elif fallback_alias and fallback_alias in inferred_roots:
        canonical = fallback_alias
    elif fallback_alias:
        canonical = fallback_alias
    else:
        canonical = project_repo

    if canonical not in identity_names:
        identity_names.append(canonical)

    return identity_names, canonical


def _add_identifier_tokens(name, all_tokens, token_to_families, extractor):
    for token in extractor.tokenize_identifier(name):
        token_low = token.lower()
        all_tokens.add(token_low)
        if token_low not in token_to_families:
            token_to_families[token_low] = []
        if len(token_to_families[token_low]) < 10 and name not in token_to_families[token_low]:
            token_to_families[token_low].append(name)


def _extend_tokens_from_level2_targets(all_tokens, token_to_families, level2_targets, extractor):
    for name in level2_targets["classes"]:
        _add_identifier_tokens(name, all_tokens, token_to_families, extractor)

    for name in level2_targets["functions"]:
        _add_identifier_tokens(name, all_tokens, token_to_families, extractor)

    for name in level2_targets["modules"]:
        for segment in name.split("."):
            _add_identifier_tokens(segment, all_tokens, token_to_families, extractor)

    for name in level2_targets["files"]:
        _add_identifier_tokens(name.rsplit(".", 1)[0], all_tokens, token_to_families, extractor)

    for name in level2_targets["directories"]:
        _add_identifier_tokens(name, all_tokens, token_to_families, extractor)


def _prune_level2_targets_for_identity(level2_targets, identity_names):
    pruned = {key: list(values) for key, values in level2_targets.items()}
    identity_names = set(identity_names)
    pruned["modules"] = [name for name in pruned["modules"] if name not in identity_names]
    pruned["directories"] = [name for name in pruned["directories"] if name not in identity_names]
    return pruned


def _drop_identity_overlaps(identity_map, namespace_symbol_map, namespace_path_map):
    if not identity_map:
        return namespace_symbol_map, namespace_path_map

    identity_keys = set(identity_map)
    namespace_symbol_map = {k: v for k, v in namespace_symbol_map.items() if k not in identity_keys}
    namespace_path_map = {k: v for k, v in namespace_path_map.items() if k not in identity_keys}
    return namespace_symbol_map, namespace_path_map


def _reconstruct_module_name(name, mapper, token_mapping, extractor):
    return ".".join(
        mapper.reconstruct_identifier(segment, token_mapping, extractor)
        for segment in name.split(".")
    )


def _bundle_variant_dir(output_dir: Path, repo_name: str, variant_index: int) -> Path:
    return output_dir / repo_name / f"variant_{variant_index}"


def _update_bundle_index(
    bundle_root: Path,
    *,
    repo_name: str,
    mapping_mode: str,
    enabled_layers: list[str],
    variant_index: int,
    seed: int,
    variant_count: int,
):
    index_path = bundle_root / "index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())

    variants = [item for item in index.get("variants", []) if item.get("variant_index") != variant_index]
    variants.append(
        {
            "name": f"variant_{variant_index}",
            "seed": seed,
            "variant_index": variant_index,
        }
    )
    variants.sort(key=lambda item: item["variant_index"])
    index.update(
        {
            "mapping_version": "l2-multi-v1",
            "repo_name": repo_name,
            "level": mapping_mode,
            "enabled_layers": enabled_layers,
            "variant_count": max(variant_count, len(variants)),
            "variants": variants,
        }
    )
    index_path.write_text(json.dumps(index, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Token-based semantic mapping for SWE-bench.")
    parser.add_argument("--instance-id", required=True, help="SWE-bench instance ID")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for mapping")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory")
    parser.add_argument("--repo-root", type=Path, default=Path("./repos"), help="Repo root")
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache"), help="Cache directory")
    parser.add_argument("--variant-index", type=int, default=0, help="Repo-level bundle variant index")
    parser.add_argument("--variant-count", type=int, default=1, help="Expected repo-level bundle variant count")
    parser.add_argument(
        "--mapping-mode",
        choices=["identity_only", "namespace_l2", "identity_namespace_l2"],
        default="identity_only",
        help="Offline bundle generation mode.",
    )
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="Backward-compatible alias for --mapping-mode identity_only.",
    )
    
    args = parser.parse_args()
    if args.identity_only:
        args.mapping_mode = "identity_only"
    project_repo = args.instance_id.split("__")[0]
    repo_path = args.repo_root / f"swe-bench_{args.instance_id}"
    
    if not repo_path.exists():
        matching = list(args.repo_root.glob(f"swe-bench_*{args.instance_id}"))
        if matching: repo_path = matching[0]
        else: return console.print(f"[red]Repo not found[/red]")

    identity_names, canonical_repo_root_name = _resolve_repo_identity_names(project_repo, repo_path)

    # 1. 提取标识符
    extractor = RepoIdentifierExtractor(repo_path)
    extractor.extract(max_workers=args.workers)
    current_identifiers = extractor.get_safe_identifiers()  # CHANGED: was get_identifiers()
    level2_targets = extractor.get_level2_namespace_targets()
    if args.mapping_mode == "identity_namespace_l2":
        level2_targets = _prune_level2_targets_for_identity(level2_targets, identity_names)
    reserved_tokens = extractor.get_reserved_tokens()

    all_current_ids = []
    for cat_ids in current_identifiers.values():
        all_current_ids.extend(cat_ids)

    # 仅从 Level 2(A) 目标提取词根，避免普通词/第三方词污染映射空间
    all_tokens = set()
    token_to_families = {}
    _extend_tokens_from_level2_targets(all_tokens, token_to_families, level2_targets, extractor)
    all_tokens = {token for token in all_tokens if token not in reserved_tokens}
    all_tokens = {token for token in all_tokens if token not in FORBIDDEN_IDENTITY_TOKENS}
    if args.mapping_mode == "identity_namespace_l2":
        for name in identity_names:
            all_tokens.discard(name.lower())
    token_to_families = {token: fam for token, fam in token_to_families.items() if token in all_tokens}
    
    console.print(f"[bold blue]Unique tokens found in repo:[/bold blue] {len(all_tokens)}")

    # 2. 全局词根缓存
    global_repo_cache = args.cache_dir / project_repo
    global_repo_cache.mkdir(parents=True, exist_ok=True)
    token_cache_file = global_repo_cache / "token_candidates.json"
    
    global_token_candidates = {}
    if token_cache_file.exists():
        with open(token_cache_file, "r") as f:
            global_token_candidates = json.load(f)
        console.print(f"[green]Loaded {len(global_token_candidates)} tokens from cache[/green]")

    # 3. 增量生成
    missing_tokens = [t for t in all_tokens if t not in global_token_candidates]
    
    # SemanticMapper calls litellm.completion directly (see glasses/mapper.py), not model.query(),
    # so default LitellmModel is only used for model_name / api_base / api_key config.
    model_config: dict = {}
    if args.api_base or args.api_key:
        model_config["model_kwargs"] = {}
        if args.api_base:
            model_config["model_kwargs"]["api_base"] = args.api_base
        if args.api_key:
            model_config["model_kwargs"]["api_key"] = args.api_key

    mapper = SemanticMapper(model_name=args.model, model_config=model_config)
    
    if missing_tokens and (args.model or args.api_base or args.api_key):
        console.print(f"[bold yellow]New tokens to map: {len(missing_tokens)}[/bold yellow]")
        # 改进：传入家族上下文，让 LLM 进行“成对/成组”生成
        new_token_candidates = mapper.generate_token_candidates(
            missing_tokens, 
            brand=project_repo, 
            context=token_to_families,
            max_workers=args.workers
        )
        global_token_candidates.update(new_token_candidates)
        with open(token_cache_file, "w") as f:
            json.dump(global_token_candidates, f, indent=2)
    elif missing_tokens:
        console.print(f"[yellow]Skipping model-backed token generation for {len(missing_tokens)} tokens[/yellow]")

    for token in all_tokens:
        global_token_candidates.setdefault(token, [token])

    # 4. 基于 Seed 生成词根映射
    # 必须把所有词根传递给 mapper 建立 token_candidates_cache 供下一步使用
    mapper.token_candidates_cache = global_token_candidates
    # 避免冲突：虚构词不能是仓库中已有的词根，也不能是系统保留词
    token_mapping = mapper.create_token_mapping(
        list(all_tokens),
        args.seed,
        avoid_tokens=all_tokens | reserved_tokens | FORBIDDEN_IDENTITY_TOKENS,
    )

    # 改进：清洗词根映射，确保“假名”也不是 Python 关键字或系统词
    stop_words = reserved_tokens | FORBIDDEN_IDENTITY_TOKENS
    clean_token_mapping = {}
    for real_t, virt_t in token_mapping.items():
        real_t_low = real_t.lower()
        # 如果真实词本身就是白名单成员，映射回它自己
        if real_t_low in stop_words:
            clean_token_mapping[real_t_low] = real_t_low
        elif virt_t.lower() in stop_words:
            # 如果假名落在白名单里，通过加后缀进行“去敏”
            clean_token_mapping[real_t_low] = f"{virt_t}_"
        else:
            clean_token_mapping[real_t_low] = virt_t
    token_mapping = clean_token_mapping
    
    # 5. 重构标识符映射 (with collision detection)
    all_real_id_set = set(all_current_ids)
    used_virtual_ids = set()
    final_mapping = {}

    for oid in all_current_ids:
        # 如果整个标识符是白名单词汇（如 'self', 'open'），不映射
        if oid.lower() in stop_words:
            continue

        virtual_id = mapper.reconstruct_identifier(oid, token_mapping, extractor)

        # Collision detection: retry up to 3 times
        retries = 0
        while (virtual_id in all_real_id_set or virtual_id in used_virtual_ids) and retries < 3:
            retries += 1
            virtual_id = mapper.reconstruct_identifier_with_offset(oid, token_mapping, extractor, seed_offset=retries)

        # Skip if still colliding
        if virtual_id in all_real_id_set or virtual_id in used_virtual_ids:
            continue

        # 兜底：如果重构后的完整标识符撞了白名单
        if virtual_id.lower() in stop_words:
            virtual_id = f"{virtual_id}_"

        # 如果重构结果没变，不存入映射表
        if virtual_id == oid:
            continue

        final_mapping[oid] = virtual_id
        used_virtual_ids.add(virtual_id)

    # 6. 保存
    instance_output = args.output_dir / args.instance_id
    instance_output.mkdir(parents=True, exist_ok=True)
    with open(instance_output / "token_mapping.json", "w") as f:
        json.dump(token_mapping, f, indent=2)
    with open(instance_output / f"mapping_seed_{args.seed}.json", "w") as f:
        json.dump(final_mapping, f, indent=2)

    bundle_root = args.output_dir / project_repo
    bundle_dir = _bundle_variant_dir(args.output_dir, project_repo, args.variant_index)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if args.mapping_mode == "identity_only":
        identity_map = {name: FIXED_REPOSITORY_ALIAS for name in identity_names}
        namespace_symbol_map = {}
        namespace_path_map = {}
    else:
        namespace_symbol_map = {
            **{
                name: mapper.reconstruct_identifier(name, token_mapping, extractor)
                for name in level2_targets["classes"]
            },
            **{
                name: mapper.reconstruct_identifier(name, token_mapping, extractor)
                for name in level2_targets["functions"]
            },
            **{
                name: _reconstruct_module_name(name, mapper, token_mapping, extractor)
                for name in level2_targets["modules"]
            },
        }
        for name in identity_names:
            namespace_symbol_map.pop(name, None)
        namespace_path_map = {
            **{
                name: mapper.reconstruct_identifier(name.rsplit(".", 1)[0], token_mapping, extractor) + ".py"
                for name in level2_targets["files"]
            },
            **{
                name: mapper.reconstruct_identifier(name, token_mapping, extractor)
                for name in level2_targets["directories"]
            },
        }
        if args.mapping_mode == "identity_namespace_l2":
            identity_map = {name: FIXED_REPOSITORY_ALIAS for name in identity_names}
        else:
            identity_map = {}
        namespace_symbol_map, namespace_path_map = _drop_identity_overlaps(
            identity_map,
            namespace_symbol_map,
            namespace_path_map,
        )

    for filename, payload in [
        ("identity_map.json", identity_map),
        ("namespace_symbol_map.json", namespace_symbol_map),
        ("namespace_path_map.json", namespace_path_map),
    ]:
        (bundle_dir / filename).write_text(json.dumps(payload, indent=2))

    enabled_layers = (
        ["namespace_l2"]
        if args.mapping_mode == "namespace_l2"
        else ["identity_l1", "namespace_l2"]
        if args.mapping_mode == "identity_namespace_l2"
        else ["identity_l1"]
    )
    (bundle_dir / "mapping_meta.json").write_text(
        json.dumps(
            {
                "mapping_version": "l2-v1",
                "repo_name": project_repo,
                "canonical_repo_root_name": canonical_repo_root_name,
                "seed": args.seed,
                "level": args.mapping_mode,
                "mapping_scope": "repo",
                "namespace_granularity": "A",
                "translated_issue_source": "re_issues/llm_django.json",
                "symbol_count": len(namespace_symbol_map),
                "path_segment_count": len(namespace_path_map),
                "enabled_layers": enabled_layers,
                "variant_index": args.variant_index,
            },
            indent=2,
        )
    )
    _update_bundle_index(
        bundle_root,
        repo_name=project_repo,
        mapping_mode=args.mapping_mode,
        enabled_layers=enabled_layers,
        variant_index=args.variant_index,
        seed=args.seed,
        variant_count=args.variant_count,
    )

    if args.mapping_mode != "identity_only":
        validate_bundle_quality(args.output_dir, project_repo, variant_name=f"variant_{args.variant_index}")

    console.print(f"[bold green]Generated token-based mapping for {args.instance_id}[/bold green]")
    console.print(f"Total Identifiers: {len(final_mapping)}")
    # 新增：输出费用
    console.print(f"FINAL_COST: {getattr(mapper.model, 'cost', 0.0):.6f}")

if __name__ == "__main__":
    main()
