#!/usr/bin/env python3

import argparse
import ast
import concurrent.futures
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "glasses"))
sys.path.insert(0, str(ROOT / "src"))

from glasses.build_verified_level3_store import (  # noqa: E402
    DEFAULT_ARROW,
    build_l3_patch,
    ensure_repo_cache,
    load_verified_dataset_map,
    pass_to_pass_clean,
    run_official_preflight,
    summarize_failure,
    write_json,
    write_overlay,
)
from glasses.level4_llm_function_body_rewrite import (  # noqa: E402
    build_level4b_runtime_record,
    extract_pass_to_pass_test_sources,
    extract_rewrite_targets_from_golden_patch,
    generate_level4b_rewrite,
    replace_function_in_source,
    rewrite_strength_gate,
    serialize_target_group,
    target_group_item_key,
    validate_rewritten_function_body,
)
from glasses.level4_agentic_function_body_rewrite import (  # noqa: E402
    FULL_TARGET_GROUP_SCOPE,
    GOLDEN_PATCH_LOCAL_SCOPE,
    MODE as AGENTIC_MODE,
    apply_agentic_submission_patch,
    build_agentic_rewrite_task,
    collect_agentic_changed_files,
    extract_agentic_changed_paths,
    get_agentic_pass_to_pass_tests,
    prepare_agentic_repo_copy,
    run_mini_agentic_rewrite,
)
from glasses.level4_semantic_rewrite import extract_patch_target_ranges  # noqa: E402

MODE = "llm_function_body_rewrite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-file", default=str(ROOT / "instances_django_mapped_merged.txt"))
    parser.add_argument("--repo-root", default=os.environ.get("LEVEL4B_REPO_ROOT", str(ROOT / "repos")))
    parser.add_argument("--output-root", default=str(ROOT / "output" / "verified_perturbations" / "level4"))
    parser.add_argument("--mode", default=MODE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-variants", type=int, default=3)
    parser.add_argument("--variant-count", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--tmp-root", default=str(ROOT / "tmp" / "verified_level4b_builder"))
    parser.add_argument(
        "--agentic-rewrite-scope",
        default=FULL_TARGET_GROUP_SCOPE,
        choices=[FULL_TARGET_GROUP_SCOPE, GOLDEN_PATCH_LOCAL_SCOPE],
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--model-config",
        default=str(ROOT / "src" / "minisweagent" / "config" / "benchmarks" / "swebench_openai_gpt54mini.yaml"),
    )
    return parser.parse_args()


def variant_dir(mode_dir: Path, index: int) -> Path:
    return mode_dir / f"variant_{index}"


def load_model_config(path: Path, model_name: str | None) -> tuple[str, dict]:
    data = yaml.safe_load(path.read_text())
    model_config = dict(data["model"]) if isinstance(data, dict) and "model" in data else dict(data)
    resolved_model_name = model_name or model_config.get("model_name")
    if resolved_model_name is None:
        raise SystemExit("model name missing from config")
    model_config["model_name"] = resolved_model_name
    return resolved_model_name, model_config


def _node_end_lineno(node):
    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is not None:
        return end_lineno
    end_lineno = getattr(node, "lineno", 0)
    for child in ast.iter_child_nodes(node):
        end_lineno = max(end_lineno, _node_end_lineno(child))
    return end_lineno


def _function_node(source: str, target_path: str, function_name: str, start_line: int):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name and node.lineno == start_line:
            return node, tree
    raise ValueError(f"function not found: {target_path}:{function_name}:{start_line}")


def _enclosing_context(source: str, node) -> str:
    tree = ast.parse(source)
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    scope = node
    while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        scope = parents.get(scope)
    start = getattr(scope, "lineno", getattr(node, "lineno", 1)) if scope is not None else getattr(node, "lineno", 1)
    end = _node_end_lineno(node)
    lines = source.splitlines(keepends=True)
    return "".join(lines[start - 1 : end])


def _rewrite_strength(original_source: str, rewritten_source: str) -> float:
    return round(1.0 - difflib.SequenceMatcher(a=original_source, b=rewritten_source).ratio(), 4)


def _candidate_signature(item: dict) -> str:
    patch = item.get("level4_patch", "")
    if not patch:
        return json.dumps(item.get("target_group", []), sort_keys=True)
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def load_existing_variant_records(mode_dir: Path) -> list[dict]:
    records = []
    for variant_path in sorted(mode_dir.glob("variant_*")):
        if not variant_path.is_dir():
            continue
        runtime_path = variant_path / "level4_runtime.json"
        status_path = variant_path / "p2p_status.json"
        if not runtime_path.exists() or not status_path.exists():
            continue
        runtime = json.loads(runtime_path.read_text())
        status = json.loads(status_path.read_text())
        records.append(
            {
                "name": variant_path.name,
                "runtime": runtime,
                "status": status,
                "signature": _candidate_signature(runtime),
            }
        )
    return records


def next_free_variant_index(mode_dir: Path) -> int:
    used = set()
    for variant_path in mode_dir.glob("variant_*"):
        if not variant_path.is_dir():
            continue
        try:
            used.add(int(variant_path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    index = 0
    while index in used:
        index += 1
    return index


def _write_variant(
    mode_dir: Path,
    slot_index: int,
    candidate: dict,
    report: dict,
    diagnostics: list[dict],
    instance_id: str,
    seed: int,
    artifacts: dict[str, str] | None = None,
) -> dict:
    slot_dir = variant_dir(mode_dir, slot_index)
    overlay_root = slot_dir / "level4_overlay"
    write_overlay(overlay_root, candidate["files"])
    runtime = build_level4b_runtime_record(
        seed=seed,
        runtime_seed=candidate["runtime_seed"],
        variant_name=slot_dir.name,
        rewritten_files=candidate["rewritten_files"],
        target_group=candidate["target_group"],
        target_group_signature=candidate["target_group_signature"],
        rewrite_summaries=candidate["rewrite_summaries"],
        behavioral_invariants=candidate["behavioral_invariants"],
        rewrite_strength=candidate["rewrite_strength"],
    )
    write_json(slot_dir / "level4_runtime.json", runtime)
    write_json(
        slot_dir / "p2p_status.json",
        {
            "instance_id": instance_id,
            "status": "verified",
            "report": report,
            "diagnostics": diagnostics,
            "artifacts": artifacts or {},
        },
    )
    (slot_dir / "level4_patch.diff").write_text(candidate["level4_patch"])
    return {
        "name": slot_dir.name,
        "runtime_seed": candidate["runtime_seed"],
        "rewritten_files": candidate["rewritten_files"],
        "target_group_signature": candidate["target_group_signature"],
        "target_group_size": candidate["target_group_size"],
        "artifacts": artifacts or {},
    }


def _load_source(repo_root: Path, relative_path: Path) -> str:
    source_path = repo_root / relative_path
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    return source_path.read_text()


def _build_group_candidate(
    *,
    repo_root: Path,
    target_group,
    seed: int,
    runtime_seed: int,
    model_name: str,
    model_config: dict,
    work_dir: Path,
    failure_feedback: str | None = None,
) -> dict:
    original_sources = {}
    rewrites_by_file = {}
    rewrite_summaries = {}
    behavioral_invariants = {}
    rewrite_strengths = []
    serialized_group = serialize_target_group(list(target_group))

    for target in target_group:
        source = original_sources.setdefault(str(target.path), _load_source(repo_root, target.path))
        node, _ = _function_node(source, str(target.path), target.function_name, target.start_line)
        original_function_source = ast.get_source_segment(source, node) or "".join(
            source.splitlines(keepends=True)[node.lineno - 1 : _node_end_lineno(node)]
        )
        enclosing_context = _enclosing_context(source, node)

        rewrite = generate_level4b_rewrite(
            model_name=model_name,
            model_config=model_config,
            path=str(target.path),
            function_name=target.function_name,
            original_function_source=original_function_source,
            enclosing_context=enclosing_context,
            failure_feedback=failure_feedback,
        )
        rewritten_function_source = rewrite["rewritten_function_source"]

        ok, reason = validate_rewritten_function_body(original_function_source, rewritten_function_source)
        if not ok:
            return {
                "status": "skipped",
                "reason": reason,
                "failed_target": target_group_item_key(
                    {
                        "path": str(target.path),
                        "function_name": target.function_name,
                        "start_line": target.start_line,
                        "end_line": target.end_line,
                    }
                ),
            }

        ok, reason = rewrite_strength_gate(original_function_source, rewritten_function_source)
        if not ok:
            return {
                "status": "skipped",
                "reason": reason,
                "failed_target": target_group_item_key(
                    {
                        "path": str(target.path),
                        "function_name": target.function_name,
                        "start_line": target.start_line,
                        "end_line": target.end_line,
                    }
                ),
            }

        rewrite_key = target_group_item_key(
            {
                "path": str(target.path),
                "function_name": target.function_name,
                "start_line": target.start_line,
                "end_line": target.end_line,
            }
        )
        rewrite_summaries[rewrite_key] = rewrite["rewrite_summary"]
        behavioral_invariants[rewrite_key] = rewrite["behavioral_invariants"]
        rewrite_strengths.append(_rewrite_strength(original_function_source, rewritten_function_source))
        rewrites_by_file.setdefault(str(target.path), []).append((target, rewritten_function_source))

    files = {}
    for path, rewrites in rewrites_by_file.items():
        rewritten_source = original_sources[path]
        for target, function_source in sorted(rewrites, key=lambda item: item[0].start_line, reverse=True):
            rewritten_source = replace_function_in_source(
                rewritten_source,
                target.function_name,
                function_source,
                target=(target.function_name, target.start_line),
            )
        try:
            ast.parse(rewritten_source)
        except SyntaxError:
            return {"status": "skipped", "reason": "syntax_error_after_replacement"}
        files[path] = rewritten_source

    level4_patch = build_l3_patch(repo_root, files, work_dir / "patch")
    return {
        "status": "ok",
        "seed": seed,
        "runtime_seed": runtime_seed,
        "target_group": serialized_group,
        "target_group_size": len(serialized_group),
        "target_group_signature": hashlib.sha256(json.dumps(serialized_group, sort_keys=True).encode("utf-8")).hexdigest(),
        "rewrite_summaries": rewrite_summaries,
        "behavioral_invariants": behavioral_invariants,
        "rewrite_strength": round(sum(rewrite_strengths) / max(1, len(rewrite_strengths)), 4),
        "rewritten_files": sorted(files),
        "files": files,
        "level4_patch": level4_patch,
    }


def _build_candidate(**kwargs):
    target = kwargs.pop("target")
    return _build_group_candidate(target_group=[target], **kwargs)


def _failure_feedback_from_local_reason(reason: str | None, failed_target: str | None = None) -> str | None:
    if not reason:
        return None
    lines = []
    if failed_target:
        lines.append(f"- target: {failed_target}")
    lines.append(f"- local failure: {reason}")
    if reason == "syntax_error":
        lines.append("- produce syntactically valid Python after the rewrite")
    elif reason == "rewrite_too_trivial":
        lines.append("- make the implementation meaningfully different instead of introducing a trivial temporary variable rewrite")
    elif reason == "syntax_error_after_replacement":
        lines.append("- keep the rewritten function compatible with its surrounding file structure and indentation")
    return "\n".join(lines)


def _read_artifact_excerpt(path: str | None, *, max_chars: int = 4000) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    text = file_path.read_text(errors="replace")
    return text[:max_chars]


def _agentic_feedback_from_local_reason(reason: str | None, failed_target: str | None, artifacts: dict[str, str]) -> dict | None:
    if not reason and not artifacts:
        return None
    feedback = {
        "local_failure_reason": reason,
        "failed_target": failed_target,
        "previous_submission_patch": _read_artifact_excerpt(artifacts.get("submission_patch"), max_chars=12000),
        "git_apply_stderr_excerpt": _read_artifact_excerpt(artifacts.get("git_apply_stderr")),
        "git_apply_stdout_excerpt": _read_artifact_excerpt(artifacts.get("git_apply_stdout")),
    }
    return {key: value for key, value in feedback.items() if value}


def _extract_test_ids_from_failure_feedback(failure_feedback: str | None) -> list[str]:
    if not failure_feedback:
        return []
    test_ids = []
    for line in failure_feedback.splitlines():
        stripped = line.strip()
        if stripped.startswith("Test: "):
            test_ids.append(stripped.split("Test: ", 1)[1].strip())
        elif stripped.startswith("- ") and "(" in stripped and ")" in stripped and "test" in stripped:
            test_ids.append(stripped[2:].strip())
    return test_ids


def _failure_feedback_from_preflight(repo_root: Path, result: dict) -> str | None:
    summary = summarize_failure(result or {})
    p2p = summary.get("pass_to_pass_failure") or []
    if not p2p:
        return None

    snippets = extract_pass_to_pass_test_sources(repo_root, p2p)
    if not snippets:
        lines = [
            "Previous attempt broke these previously passing tests. The next rewrite must keep them passing.",
            "Tests:",
        ]
        lines.extend(f"- {name}" for name in p2p)
        return "\n".join(lines)

    lines = ["Previous attempt broke these previously passing tests. The next rewrite must keep them passing."]
    for snippet in snippets:
        lines.append(f"Test: {snippet['test_id']}")
        lines.append(f"Path: {snippet['path']}")
        lines.append("```python")
        lines.append((snippet.get("source") or "").rstrip("\n"))
        lines.append("```")
    unresolved = [name for name in p2p if name not in {item["test_id"] for item in snippets}]
    if unresolved:
        lines.append("Tests without recovered source:")
        lines.extend(f"- {name}" for name in unresolved)
    return "\n".join(lines)


def _agentic_feedback_from_preflight(result: dict, artifacts: dict[str, str]) -> dict:
    summary = summarize_failure(result or {})
    return {
        "preflight_summary": summary,
        "pass_to_pass_failure": summary.get("pass_to_pass_failure") or [],
        "fail_to_pass_failure": summary.get("fail_to_pass_failure") or [],
        "previous_submission_patch": _read_artifact_excerpt(artifacts.get("submission_patch"), max_chars=12000),
        "preflight_report_excerpt": _read_artifact_excerpt(artifacts.get("preflight_report")),
        "preflight_test_output_excerpt": _read_artifact_excerpt(artifacts.get("preflight_test_output")),
    }


def _build_target_sources(repo_root: Path, target_group) -> dict[str, dict]:
    target_sources = {}
    source_cache = {}
    for target in target_group:
        source = source_cache.setdefault(str(target.path), _load_source(repo_root, target.path))
        node, _ = _function_node(source, str(target.path), target.function_name, target.start_line)
        original_function_source = ast.get_source_segment(source, node) or "".join(
            source.splitlines(keepends=True)[node.lineno - 1 : _node_end_lineno(node)]
        )
        target_sources[
            target_group_item_key(
                {
                    "path": str(target.path),
                    "function_name": target.function_name,
                    "start_line": target.start_line,
                    "end_line": target.end_line,
                }
            )
        ] = {
            "original_function_source": original_function_source,
            "enclosing_context": _enclosing_context(source, node),
        }
    return target_sources


def _build_patch_local_context(instance: dict, repo_root: Path, *, radius: int = 30) -> dict[str, str]:
    ranges_by_path = extract_patch_target_ranges(instance.get("patch", ""))
    context = {}
    for path, ranges in ranges_by_path.items():
        source_path = repo_root / path
        if not source_path.exists():
            continue
        lines = source_path.read_text().splitlines(keepends=True)
        chunks = []
        for start, end in ranges:
            lo = max(1, start - radius)
            hi = min(len(lines), end + radius)
            snippet = "".join(lines[lo - 1 : hi])
            chunks.append(f"@@ local context {start}-{end} @@\n{snippet}")
        if chunks:
            context[path] = "\n".join(chunks)
    return context


def _agentic_rewrite_strength(repo_root: Path, target_group, changed_files: dict[str, str]) -> float | None:
    strengths = []
    source_cache = {}
    for target in target_group:
        path = str(target.path)
        if path not in changed_files:
            continue
        original_source = source_cache.setdefault(path, _load_source(repo_root, target.path))
        rewritten_source = changed_files[path]
        original_node, _ = _function_node(original_source, path, target.function_name, target.start_line)
        try:
            rewritten_node, _ = _function_node(rewritten_source, path, target.function_name, target.start_line)
        except ValueError:
            continue
        original_segment = ast.get_source_segment(original_source, original_node) or "".join(
            original_source.splitlines(keepends=True)[original_node.lineno - 1 : _node_end_lineno(original_node)]
        )
        rewritten_segment = ast.get_source_segment(rewritten_source, rewritten_node) or "".join(
            rewritten_source.splitlines(keepends=True)[rewritten_node.lineno - 1 : _node_end_lineno(rewritten_node)]
        )
        strengths.append(_rewrite_strength(original_segment, rewritten_segment))
    if not strengths:
        return None
    return round(sum(strengths) / len(strengths), 4)


def _attempt_artifacts(work_dir: Path) -> dict[str, str]:
    artifacts = {}
    mapping = {
        "submission_patch": work_dir / "submission.patch",
        "git_apply_stdout": work_dir / "git_apply_stdout.txt",
        "git_apply_stderr": work_dir / "git_apply_stderr.txt",
        "preflight_report": work_dir / "preflight" / "report.json",
        "preflight_test_output": work_dir / "preflight" / "test_output.txt",
        "trajectory": work_dir / "agent.traj.json",
    }
    for key, path in mapping.items():
        if path.exists():
            artifacts[key] = str(path)
    return artifacts


def _build_agentic_group_candidate(
    *,
    instance: dict,
    instance_id: str,
    repo_root: Path,
    target_group,
    seed: int,
    runtime_seed: int,
    model_name: str,
    model_config: dict,
    work_dir: Path,
    failure_feedback: dict | None = None,
    rewrite_scope: str = FULL_TARGET_GROUP_SCOPE,
) -> dict:
    serialized_group = serialize_target_group(list(target_group))
    target_sources = _build_target_sources(repo_root, target_group)
    golden_patch_local_context = None
    if rewrite_scope == GOLDEN_PATCH_LOCAL_SCOPE:
        golden_patch_local_context = _build_patch_local_context(instance, repo_root)
    task = build_agentic_rewrite_task(
        instance_id=instance_id,
        target_group=serialized_group,
        target_sources=target_sources,
        pass_to_pass_tests=get_agentic_pass_to_pass_tests(instance),
        rewrite_scope=rewrite_scope,
        golden_patch_local_context=golden_patch_local_context,
        previous_attempt_feedback=failure_feedback,
    )
    agent_repo_root = prepare_agentic_repo_copy(repo_root, work_dir)
    run_output = run_mini_agentic_rewrite(
        instance=instance,
        repo_root=agent_repo_root,
        task=task,
        model_name=model_name,
        output_path=work_dir / "agent.traj.json",
        config_path=ROOT / "src" / "minisweagent" / "config" / "benchmarks" / "swebench.yaml",
    )
    submission = run_output.get("submission", "")
    if not submission.strip():
        return {"status": "skipped", "reason": "empty_agentic_submission"}
    changed_paths = extract_agentic_changed_paths(submission)
    if not changed_paths:
        return {"status": "skipped", "reason": "no_agentic_changes"}
    apply_agentic_submission_patch(agent_repo_root, submission, work_dir / "submission.patch")
    changed_files = collect_agentic_changed_files(agent_repo_root, changed_paths)
    if not changed_files:
        return {"status": "skipped", "reason": "no_agentic_changes"}
    level4_patch = build_l3_patch(repo_root, changed_files, work_dir / "patch")
    rewrite_summaries = run_output.get("rewrite_summaries") or {
        target_group_item_key(item): "agentic rewrite"
        for item in serialized_group
    }
    behavioral_invariants = run_output.get("behavioral_invariants") or {
        target_group_item_key(item): ["same behavior as original implementation"]
        for item in serialized_group
    }
    return {
        "status": "ok",
        "seed": seed,
        "runtime_seed": runtime_seed,
        "target_group": serialized_group,
        "target_group_size": len(serialized_group),
        "target_group_signature": hashlib.sha256(json.dumps(serialized_group, sort_keys=True).encode("utf-8")).hexdigest(),
        "rewrite_summaries": rewrite_summaries,
        "behavioral_invariants": behavioral_invariants,
        "rewrite_strength": _agentic_rewrite_strength(repo_root, target_group, changed_files),
        "rewritten_files": sorted(changed_files),
        "files": changed_files,
        "level4_patch": level4_patch,
        "generator_backend": "mini_swe_agent",
    }


def _classify_skipped_reason(diagnostics: list[dict]) -> str:
    has_local = False
    has_p2p = False
    local_reasons = set()
    for item in diagnostics:
        reason = item.get("reason")
        if reason:
            has_local = True
            local_reasons.add(reason)
        result = item.get("result") or {}
        if result.get("pass_to_pass_failure"):
            has_p2p = True
    if has_local and has_p2p:
        return "mixed_local_gate_and_p2p_fail"
    if has_p2p:
        return "p2p_only"
    if has_local:
        if local_reasons == {"syntax_error", "syntax_error_after_replacement"} or local_reasons == {"syntax_error"} or local_reasons == {"syntax_error_after_replacement"}:
            return "local_gate_only_syntax"
        if local_reasons == {"rewrite_too_trivial"}:
            return "local_gate_only_trivial"
        return "local_gate_only_mixed"
    return "p2p_failed_all_variants"


def verify_instance(
    instance: dict,
    repo_root: Path,
    mode_dir: Path,
    tmp_root: Path,
    *,
    seed: int,
    variant_count: int,
    max_attempts: int,
    target_variants: int,
    redo: bool,
    model_name: str,
    model_config: dict,
    mode: str = MODE,
    agentic_rewrite_scope: str = FULL_TARGET_GROUP_SCOPE,
) -> dict:
    instance_id = instance["instance_id"]
    print(f"{instance_id} start level4b verification", flush=True)
    if redo and mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)

    existing = [] if redo else load_existing_variant_records(mode_dir)
    chosen = [
        {
            "name": item["name"],
            "runtime_seed": item["runtime"].get("runtime_seed"),
            "rewritten_files": item["runtime"].get("rewritten_files", []),
            "target_group_signature": item["runtime"].get("target_group_signature"),
            "target_group_size": item["runtime"].get("target_group_size"),
        }
        for item in existing
        if item["status"].get("status") == "verified"
    ]
    seen = {item["signature"] for item in existing}
    if len(chosen) >= target_variants:
        return {"instance_id": instance_id, "status": "verified", "mode": mode, "variants": chosen[:target_variants]}

    target_group, skip_reason = extract_rewrite_targets_from_golden_patch(instance.get("patch", ""), repo_root)
    if skip_reason:
        write_json(mode_dir / "level4_runtime.json", {"mode": mode, "status": "skipped", "reason": skip_reason, "seed": seed})
        write_json(mode_dir / "p2p_status.json", {"instance_id": instance_id, "status": "skipped", "reason": skip_reason})
        return {"instance_id": instance_id, "status": "skipped", "reason": skip_reason}

    diagnostics = []
    failure_feedback = None
    if mode == AGENTIC_MODE:
        attempt_budget = 3
        candidate_builder = _build_agentic_group_candidate
        forced_model_config = dict(model_config)
        forced_kwargs = dict(forced_model_config.get("model_kwargs", {}))
        forced_kwargs["temperature"] = 1
        forced_model_config["model_kwargs"] = forced_kwargs
    else:
        attempt_budget = max_attempts
        candidate_builder = _build_group_candidate
        forced_model_config = model_config

    for attempt_index in range(attempt_budget):
        if len(chosen) >= target_variants:
            break
        runtime_seed = seed + attempt_index
        work_dir = tmp_root / instance_id / f"attempt_{attempt_index}"
        try:
            if mode == AGENTIC_MODE:
                candidate = candidate_builder(
                    instance=instance,
                    instance_id=instance_id,
                    repo_root=repo_root,
                    target_group=target_group,
                    seed=seed,
                    runtime_seed=runtime_seed,
                    model_name=model_name,
                    model_config=forced_model_config,
                    work_dir=work_dir,
                    failure_feedback=failure_feedback,
                    rewrite_scope=agentic_rewrite_scope,
                )
            else:
                candidate = candidate_builder(
                    repo_root=repo_root,
                    target_group=target_group,
                    seed=seed,
                    runtime_seed=runtime_seed,
                    model_name=model_name,
                    model_config=forced_model_config,
                    work_dir=work_dir,
                    failure_feedback=failure_feedback,
                )
        except Exception as exc:
            status = {"instance_id": instance_id, "status": "error", "error": str(exc)}
            artifacts = _attempt_artifacts(work_dir)
            if artifacts:
                status["artifacts"] = artifacts
            write_json(mode_dir / "level4_runtime.json", {"mode": mode, "status": "error", "reason": str(exc), "seed": seed})
            write_json(mode_dir / "p2p_status.json", status)
            return status
        if candidate.get("status") != "ok":
            artifacts = _attempt_artifacts(work_dir)
            diagnostics.append(
                {
                    "attempt_index": attempt_index,
                    "runtime_seed": runtime_seed,
                    "target_group_signature": candidate.get("target_group_signature"),
                    "status": candidate.get("status"),
                    "reason": candidate.get("reason"),
                    "artifacts": artifacts,
                }
            )
            if mode == AGENTIC_MODE:
                failure_feedback = _agentic_feedback_from_local_reason(
                    candidate.get("reason"),
                    candidate.get("failed_target"),
                    artifacts,
                )
            else:
                failure_feedback = _failure_feedback_from_local_reason(candidate.get("reason"), candidate.get("failed_target"))
            continue

        signature = _candidate_signature(candidate)
        if signature in seen:
            diagnostics.append(
                {
                    "attempt_index": attempt_index,
                    "runtime_seed": runtime_seed,
                    "target_group_signature": candidate["target_group_signature"],
                    "status": "duplicate_signature",
                }
            )
            continue

        slot_index = next_free_variant_index(mode_dir)
        print(
            f"{instance_id} preflight variant={slot_index} seed={runtime_seed} files={','.join(candidate['rewritten_files'])}",
            flush=True,
        )
        overlay_root = work_dir / "overlay"
        write_overlay(overlay_root, candidate["files"])
        log_dir = work_dir / "preflight"
        log_dir.mkdir(parents=True, exist_ok=True)
        result = run_official_preflight(
            instance,
            overlay_root,
            candidate["rewritten_files"],
            candidate["level4_patch"],
            log_dir,
            timeout=None,
        )
        diagnostics.append(
            {
                "attempt_index": attempt_index,
                "runtime_seed": runtime_seed,
                "target_group_signature": candidate["target_group_signature"],
                "rewritten_files": candidate["rewritten_files"],
                "result": summarize_failure(result),
                "artifacts": _attempt_artifacts(work_dir),
            }
        )
        if pass_to_pass_clean(result):
            candidate["runtime_seed"] = runtime_seed
            artifacts = _attempt_artifacts(work_dir)
            chosen.append(
                _write_variant(
                    mode_dir,
                    slot_index,
                    candidate,
                    result["report"],
                    diagnostics,
                    instance_id,
                    seed,
                    artifacts=artifacts,
                )
            )
            seen.add(signature)
            print(f"{instance_id} verified variant_{slot_index}", flush=True)
            failure_feedback = None
        else:
            artifacts = _attempt_artifacts(work_dir)
            if mode == AGENTIC_MODE:
                failure_feedback = _agentic_feedback_from_preflight(result, artifacts)
            else:
                failure_feedback = _failure_feedback_from_preflight(repo_root, result)

    if chosen:
        if len(chosen) >= target_variants:
            return {"instance_id": instance_id, "status": "verified", "mode": mode, "variants": chosen[:target_variants]}

        runtime = {
            "mode": mode,
            "status": "partial_verified",
            "reason": "exhausted_attempts",
            "seed": seed,
            "verified_count": len(chosen),
            "target_variants": target_variants,
            "max_attempts": attempt_budget,
        }
        write_json(mode_dir / "level4_runtime.json", runtime)
        write_json(
            mode_dir / "p2p_status.json",
            {
                "instance_id": instance_id,
                "status": "partial_verified",
                "reason": "exhausted_attempts",
                "verified_count": len(chosen),
                "target_variants": target_variants,
                "max_attempts": attempt_budget,
                "diagnostics": diagnostics,
            },
        )
        return {
            "instance_id": instance_id,
            "status": "partial_verified",
            "reason": "exhausted_attempts",
            "mode": mode,
            "verified_count": len(chosen),
            "target_variants": target_variants,
            "variants": chosen,
            "diagnostics": diagnostics,
        }

    skipped_reason = _classify_skipped_reason(diagnostics)
    runtime = {"mode": mode, "status": "skipped", "reason": skipped_reason, "seed": seed}
    write_json(mode_dir / "level4_runtime.json", runtime)
    write_json(
        mode_dir / "p2p_status.json",
        {
            "instance_id": instance_id,
            "status": "skipped",
            "reason": skipped_reason,
            "diagnostics": diagnostics,
        },
    )
    return {"instance_id": instance_id, "status": "skipped", "reason": skipped_reason, "diagnostics": diagnostics}


def main() -> None:
    args = parse_args()
    if args.target_variants < 1:
        raise SystemExit("--target-variants must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    filter_ids = [line.strip() for line in Path(args.filter_file).read_text().splitlines() if line.strip()]
    if args.limit:
        filter_ids = filter_ids[: args.limit]

    output_root = Path(args.output_root)
    tmp_root = Path(args.tmp_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    instances = load_verified_dataset_map(filter_ids, DEFAULT_ARROW)
    model_name, model_config = load_model_config(Path(args.model_config), args.model_name)
    index_path = output_root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index_lock = threading.Lock()

    def run_one(instance_id: str) -> dict:
        instance = instances[instance_id]
        mode_dir = output_root / instance_id / args.mode
        repo_root = ensure_repo_cache(instance, Path(args.repo_root) / instance_id)
        return verify_instance(
        instance,
        repo_root,
        mode_dir,
        tmp_root,
        seed=args.seed,
            variant_count=args.variant_count,
            max_attempts=args.max_attempts,
            target_variants=args.target_variants,
        redo=args.redo,
        model_name=model_name,
        model_config=model_config,
        mode=args.mode,
        agentic_rewrite_scope=args.agentic_rewrite_scope,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {executor.submit(run_one, instance_id): instance_id for instance_id in filter_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            instance_id = future_to_id[future]
            try:
                status = future.result()
            except Exception as exc:
                status = {"instance_id": instance_id, "status": "error", "error": str(exc)}
            with index_lock:
                index[instance_id] = status
                write_json(index_path, index)
            print(instance_id, status["status"], status.get("reason", len(status.get("variants", []))), flush=True)


if __name__ == "__main__":
    main()
