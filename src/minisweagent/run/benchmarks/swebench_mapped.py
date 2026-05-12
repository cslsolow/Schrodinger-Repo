#!/usr/bin/env python3

"""Run mini-SWE-agent with semantic mapping on SWE-bench instances."""

import concurrent.futures
import json
import logging
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import shutil
from pathlib import Path

import typer
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.mapped_agent import SemanticMappingAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    DEFAULT_CONFIG_FILE,
    apply_hf_offline_env_for_datasets,
    datasets_local_files_only,
    filter_instances,
    get_sb_environment,
    load_swebench_instances,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

# Add glasses to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "glasses"))
from manager import SemanticMappingManager
from intra_file_reorder import (
    build_intra_file_reorder_plan,
    materialize_intra_file_variant,
    restore_original_file_order,
)
from layout_planner import resolve_instance_repo_root
from mapping_bundle import select_mapping_bundle_dir as _select_mapping_bundle_dir

_HELP_TEXT = """Run mini-SWE-agent with semantic mapping on SWEBench instances."""
LEVEL4_VERIFIED_MODES = {"llm_function_body_rewrite", "agentic_function_body_rewrite"}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


class MappedProgressTrackingAgent(SemanticMappingAgent):
    """SemanticMappingAgent with progress tracking."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


def rename_repo_in_container(env: Environment, real_name: str, virtual_name: str) -> bool:
    """Expose a stable virtual alias without moving the tracked repo root."""
    fixed_repo_roots = {
        "mwaskom": "/testbed/seaborn",
        "pallets": "/testbed/src/flask",
        "psf": "/testbed/requests",
        "pydata": "/testbed/xarray",
        "pylint-dev": "/testbed/pylint",
        "pytest-dev": "/testbed/src",
        "scikit-learn": "/testbed/sklearn",
        "sphinx-doc": "/testbed/sphinx",
    }
    candidate_paths = [
        fixed_repo_roots.get(real_name),
        f"/testbed/{real_name}",
        f"/testbed/lib/{real_name}",
        f"/testbed/src/{real_name}",
    ]
    source_path = None
    last_result = None
    for candidate in candidate_paths:
        if not candidate:
            continue
        result = env.execute({"command": f"test -e {candidate}"})
        last_result = result
        if result["returncode"] == 0:
            source_path = candidate
            break
    if source_path is None:
        logger.warning(f"Failed to rename repo: {last_result}")
        return False

    result = env.execute({"command": f"rm -rf /testbed/{virtual_name} && ln -s {source_path} /testbed/{virtual_name}"})
    if result["returncode"] != 0:
        logger.warning(f"Failed to create repo alias symlink: {result}")
        return False
    return True


def resolve_mapping_bundle_dir(semantic_mappings: Path, instance_id: str) -> Path:
    project_name = instance_id.split("__", 1)[0]
    return semantic_mappings / project_name


def select_mapping_bundle_dir(bundle_root: Path, semantic_seed: int) -> tuple[Path, dict | None]:
    return _select_mapping_bundle_dir(bundle_root, semantic_seed)


def build_mapping_info(
    *,
    mapping_manager: SemanticMappingManager | None,
    bundle_dir: Path | None,
    bundle_root_dir: Path | None = None,
    enabled_layers: list[str] | None,
    semantic_seed: int,
    project_name: str | None,
    bundle_variant: dict | None = None,
) -> dict:
    if mapping_manager is None or bundle_dir is None:
        return {}
    info = {
        "mapping": {
            "bundle_dir": str(bundle_dir),
            "bundle_metadata": mapping_manager.bundle.get("mapping_meta", {}) if mapping_manager.bundle else {},
            "enabled_layers": list(mapping_manager.enabled_layers),
            "semantic_seed": semantic_seed,
            "project_name": project_name,
        }
    }
    if bundle_root_dir is not None and bundle_root_dir != bundle_dir:
        info["mapping"]["bundle_root_dir"] = str(bundle_root_dir)
    if bundle_variant is not None:
        info["mapping"]["bundle_variant"] = bundle_variant
    return info


def resolve_verified_level3_mode_dir(level3_verified_store: Path, instance_id: str, level3_mode: str) -> Path:
    return level3_verified_store / instance_id / level3_mode


def resolve_verified_level4_mode_dir(level4_verified_store: Path, instance_id: str, level4_mode: str) -> Path:
    return level4_verified_store / instance_id / level4_mode


def load_verified_level3_record(level3_verified_store: Path, instance_id: str, level3_mode: str) -> dict | None:
    index_path = level3_verified_store / "index.json"
    if not index_path.exists():
        return None
    data = json.loads(index_path.read_text())
    record = data.get(instance_id)
    if not record:
        return None
    if record.get("status") != "verified":
        return record
    mode_dir = resolve_verified_level3_mode_dir(level3_verified_store, instance_id, level3_mode)
    variants = record.get("variants", [])
    if variants:
        return {
            "status": "verified",
            "variants": variants,
            "mode_dir": str(mode_dir),
        }
    return record


def load_verified_level4_record(level4_verified_store: Path, instance_id: str, level4_mode: str) -> dict | None:
    index_path = level4_verified_store / "index.json"
    if not index_path.exists():
        return None
    data = json.loads(index_path.read_text())
    record = data.get(instance_id)
    if not record:
        return None
    if record.get("status") not in {"verified", "partial_verified"}:
        return record
    variants = record.get("variants", [])
    if variants:
        mode_dir = record.get("mode_dir") or resolve_verified_level4_mode_dir(level4_verified_store, instance_id, level4_mode)
        return {
            "status": record.get("status", "verified"),
            "variants": variants,
            "mode_dir": str(mode_dir),
        }
    return record


def select_verified_level3_variant(record: dict, *, instance_id: str, level3_seed: int) -> dict | None:
    variants = record.get("variants", [])
    if not variants:
        return None
    index = random.Random(f"{instance_id}:{level3_seed}:verified_level3").randrange(len(variants))
    return variants[index]


def select_verified_level4_variant(record: dict, *, instance_id: str, level4_seed: int) -> dict | None:
    variants = record.get("variants", [])
    if not variants:
        return None
    index = random.Random(f"{instance_id}:{level4_seed}:verified_level4").randrange(len(variants))
    return variants[index]


def copy_text_to_container(container_id: str, executable: str, relative_path: str, content: str, *, tmp_root: Path) -> None:
    tmp_root.mkdir(parents=True, exist_ok=True)
    local_path = tmp_root / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(content)
    subprocess.run(
        [executable, "exec", container_id, "mkdir", "-p", str(Path("/testbed") / relative_path).rsplit("/", 1)[0]],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run([executable, "cp", str(local_path), f"{container_id}:/testbed/{relative_path}"], check=True)


def extract_changed_patch_paths(submission: str) -> list[str]:
    paths = []
    seen = set()
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+)$", submission, re.MULTILINE):
        old_path, new_path = match.groups()
        path = new_path if new_path != "/dev/null" else old_path
        if path == "/dev/null" or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def extract_changed_repo_paths(status_output: str) -> list[str]:
    paths = []
    seen = set()
    for line in status_output.splitlines():
        if not line:
            continue
        entry = line[3:]
        path = entry.split(" -> ", 1)[1] if " -> " in entry else entry
        path = path.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def copy_file_from_container(container_id: str, executable: str, relative_path: str, *, tmp_root: Path) -> bytes | None:
    tmp_root.mkdir(parents=True, exist_ok=True)
    local_path = tmp_root / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [executable, "cp", f"{container_id}:/testbed/{relative_path}", str(local_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return local_path.read_bytes()


def copy_head_file_from_container(container_id: str, executable: str, relative_path: str) -> bytes | None:
    result = subprocess.run(
        [executable, "exec", container_id, "git", "-C", "/testbed", "show", f"HEAD:{relative_path}"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def build_git_patch_from_bytes(base_bytes: bytes | None, new_bytes: bytes | None, relative_path: str, *, tmp_root: Path) -> str:
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        td_path = Path(td)
        old_path = td_path / "orig" / relative_path
        new_path = td_path / "new" / relative_path
        old_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if base_bytes is not None:
            old_path.write_bytes(base_bytes)
        if new_bytes is not None:
            new_path.write_bytes(new_bytes)
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--binary",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                str(Path("orig") / relative_path),
                str(Path("new") / relative_path),
            ],
            capture_output=True,
            text=True,
            cwd=td_path,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr or result.stdout)
        patch = result.stdout
    patch = patch.replace(f"a/orig/{relative_path}", f"a/{relative_path}")
    patch = patch.replace(f"b/new/{relative_path}", f"b/{relative_path}")
    return patch


def build_state_based_submissions_from_contents(
    *,
    base_repo_root: Path | None = None,
    base_file_bytes: dict[str, bytes | None] | None = None,
    perturbed_file_bytes: dict[str, bytes | None],
    current_file_bytes: dict[str, bytes | None],
    file_orders: dict,
    level3_submission_mode: str,
    tmp_root: Path,
) -> tuple[str, str]:
    agent_patches = []
    final_patches = []
    for relative_path in current_file_bytes:
        if base_file_bytes is not None:
            base_bytes = base_file_bytes.get(relative_path)
        else:
            if base_repo_root is None:
                raise ValueError("build_state_based_submissions_from_contents requires base_repo_root or base_file_bytes")
            base_path = base_repo_root / relative_path
            base_bytes = base_path.read_bytes() if base_path.exists() else None
        perturbed_bytes = perturbed_file_bytes.get(relative_path, base_bytes)
        current_bytes = current_file_bytes.get(relative_path)

        agent_patch = build_git_patch_from_bytes(perturbed_bytes, current_bytes, relative_path, tmp_root=tmp_root / "agent")
        if agent_patch:
            agent_patches.append(agent_patch)

        final_bytes = current_bytes
        if (
            level3_submission_mode == "remap"
            and current_bytes is not None
            and relative_path in file_orders
        ):
            restored = restore_original_file_order(
                current_bytes.decode("utf-8", errors="surrogateescape"),
                file_orders[relative_path],
            )
            final_bytes = restored.encode("utf-8", errors="surrogateescape")

        final_patch = build_git_patch_from_bytes(base_bytes, final_bytes, relative_path, tmp_root=tmp_root / "final")
        if final_patch:
            final_patches.append(final_patch)

    return "".join(agent_patches), "".join(final_patches)


def remap_intra_file_submission_from_contents(
    submission: str,
    *,
    base_repo_root: Path,
    file_orders: dict,
    current_file_bytes: dict[str, bytes | None],
    tmp_root: Path,
) -> str:
    if not submission.strip():
        return submission
    changed_paths = extract_changed_patch_paths(submission)
    if not changed_paths:
        return submission

    remapped = []
    for relative_path in changed_paths:
        base_path = base_repo_root / relative_path
        base_bytes = base_path.read_bytes() if base_path.exists() else None
        current_bytes = current_file_bytes.get(relative_path)
        if current_bytes is not None and relative_path in file_orders:
            restored = restore_original_file_order(
                current_bytes.decode("utf-8", errors="surrogateescape"),
                file_orders[relative_path],
            )
            current_bytes = restored.encode("utf-8", errors="surrogateescape")
        patch = build_git_patch_from_bytes(base_bytes, current_bytes, relative_path, tmp_root=tmp_root)
        if patch:
            remapped.append(patch)
    return "".join(remapped)


def maybe_remap_level3_submission(
    submission: str,
    *,
    instance_id: str,
    env: Environment,
    output_dir: Path,
    level3_mode: str,
    level3_repo_root: Path | None,
    level3_info: dict | None,
) -> str:
    if level3_mode != "intra_file_reorder" or not submission.strip():
        return submission
    if not level3_info or level3_info.get("status") != "applied":
        return submission
    if level3_repo_root is None:
        return submission
    if not hasattr(env, "container_id") or not hasattr(env, "config"):
        return submission

    changed_paths = extract_changed_patch_paths(submission)
    if not changed_paths:
        return submission

    base_repo_root = resolve_instance_repo_root(level3_repo_root, instance_id)
    tmp_root = output_dir / instance_id / "level3_remap"
    current_file_bytes = {
        path: copy_file_from_container(env.container_id, env.config.executable, path, tmp_root=tmp_root / "container")
        for path in changed_paths
    }
    return remap_intra_file_submission_from_contents(
        submission,
        base_repo_root=base_repo_root,
        file_orders=level3_info.get("file_orders", {}),
        current_file_bytes=current_file_bytes,
        tmp_root=tmp_root / "diff",
    )


def maybe_build_state_based_submissions(
    *,
    instance_id: str,
    env: Environment,
    output_dir: Path,
    level3_mode: str,
    level3_repo_root: Path | None,
    level3_info: dict | None,
    level3_submission_mode: str,
    level4_mode: str = "",
    level4_repo_root: Path | None = None,
    level4_info: dict | None = None,
) -> dict[str, str] | None:
    if not hasattr(env, "container_id") or not hasattr(env, "config"):
        return None

    status_result = env.execute({"command": "git -C /testbed status --porcelain"})
    if status_result.get("returncode") != 0:
        return None
    changed_paths = extract_changed_repo_paths(status_result.get("output", ""))
    if not changed_paths:
        return {"agent_submission": "", "final_submission": ""}

    overlay_root = output_dir / instance_id / "level3_overlay"
    level4_overlay_root = output_dir / instance_id / "level4_overlay"
    tmp_root = output_dir / instance_id / "level3_submission_state"

    base_file_bytes = {
        path: copy_head_file_from_container(env.container_id, env.config.executable, path)
        for path in changed_paths
    }
    current_file_bytes = {
        path: copy_file_from_container(env.container_id, env.config.executable, path, tmp_root=tmp_root / "container")
        for path in changed_paths
    }
    perturbed_file_bytes = {}
    for path in changed_paths:
        level4_overlay_path = level4_overlay_root / path
        overlay_path = overlay_root / path
        if level4_overlay_path.exists():
            perturbed_file_bytes[path] = level4_overlay_path.read_bytes()
        elif overlay_path.exists():
            perturbed_file_bytes[path] = overlay_path.read_bytes()
        else:
            perturbed_file_bytes[path] = base_file_bytes.get(path)

    agent_submission, final_submission = build_state_based_submissions_from_contents(
        base_file_bytes=base_file_bytes,
        perturbed_file_bytes=perturbed_file_bytes,
        current_file_bytes=current_file_bytes,
        file_orders=(level3_info or {}).get("file_orders", {}),
        level3_submission_mode=level3_submission_mode,
        tmp_root=tmp_root / "diff",
    )
    return {
        "agent_submission": agent_submission,
        "final_submission": final_submission,
    }


def maybe_remap_mapping_submission(submission: str, mapping_manager: SemanticMappingManager | None) -> str:
    if not submission.strip() or mapping_manager is None:
        return submission
    return mapping_manager.to_real_patch(submission)


def maybe_remap_submission(
    submission: str,
    *,
    instance_id: str,
    env: Environment,
    output_dir: Path,
    mapping_manager: SemanticMappingManager | None,
    level3_mode: str,
    level3_repo_root: Path | None,
    level3_info: dict | None,
    level3_submission_mode: str,
) -> str:
    submission = maybe_remap_mapping_submission(submission, mapping_manager)
    if level3_submission_mode == "remap":
        submission = maybe_remap_level3_submission(
            submission,
            instance_id=instance_id,
            env=env,
            output_dir=output_dir,
            level3_mode=level3_mode,
            level3_repo_root=level3_repo_root,
            level3_info=level3_info,
        )
    return submission


def maybe_apply_level3_intra_file(
    *,
    instance: dict,
    env: Environment,
    output_dir: Path,
    level3_mode: str,
    level3_trajectories: Path | None,
    level3_repo_root: Path | None,
    level3_verified_store: Path | None,
    level3_seed: int,
) -> dict:
    if level3_mode != "intra_file_reorder":
        return {}
    if not hasattr(env, "container_id") or not hasattr(env, "config"):
        raise ValueError("Level 3 intra-file reorder currently requires a docker-backed environment")

    instance_id = instance["instance_id"]
    if level3_verified_store is not None:
        record = load_verified_level3_record(level3_verified_store, instance_id, level3_mode)
        if record:
            if record.get("status") == "skipped":
                return {"level3": {"mode": level3_mode, "status": "skipped", "reason": record.get("reason", "verified_store_skipped")}}
            variant = select_verified_level3_variant(record, instance_id=instance_id, level3_seed=level3_seed)
            if variant:
                mode_dir = Path(record["mode_dir"])
                variant_name = variant["name"]
                variant_dir = mode_dir / variant_name
                overlay_root = variant_dir / "level3_overlay"
                runtime_path = variant_dir / "level3_runtime.json"
                runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
                tmp_root = output_dir / instance_id / "level3_overlay"
                for overlay_file in overlay_root.rglob("*"):
                    if overlay_file.is_file():
                        relative_path = overlay_file.relative_to(overlay_root).as_posix()
                        copy_text_to_container(
                            env.container_id,
                            env.config.executable,
                            relative_path,
                            overlay_file.read_text(),
                            tmp_root=tmp_root,
                        )

                level3_info = {
                    "mode": level3_mode,
                    "status": runtime.get("status", "applied"),
                    "seed": level3_seed,
                    "variant_index": runtime.get("variant_index"),
                    "runtime_seed": runtime.get("runtime_seed"),
                    "variant_name": variant_name,
                    "rewritten_files": runtime.get("rewritten_files", []),
                    "file_orders": runtime.get("file_orders", {}),
                    "verified_store": str(level3_verified_store),
                    "active_files": variant.get("active_files", []),
                }
                (output_dir / instance_id / "level3_runtime.json").parent.mkdir(parents=True, exist_ok=True)
                (output_dir / instance_id / "level3_runtime.json").write_text(json.dumps(level3_info, indent=2))
                return {"level3": level3_info}

    if level3_trajectories is None or level3_repo_root is None:
        raise ValueError(
            "Level 3 intra-file reorder requires either --level3-verified-store or both --level3-trajectories and --level3-repo-root"
        )

    traj_path = level3_trajectories / instance_id / f"{instance_id}.traj.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Level 3 trajectory not found: {traj_path}")

    repo_src = resolve_instance_repo_root(level3_repo_root, instance_id)
    plan = build_intra_file_reorder_plan(traj_path, repo_src, seed=level3_seed)
    materialized = materialize_intra_file_variant(repo_src, plan, runtime_seed=level3_seed)
    if materialized.get("status") != "ok":
        return {"level3": {"mode": level3_mode, "status": materialized.get("status", "skipped")}}

    tmp_root = output_dir / instance_id / "level3_overlay"
    for relative_path, content in materialized["files"].items():
        copy_text_to_container(env.container_id, env.config.executable, relative_path, content, tmp_root=tmp_root)

    level3_info = {
        "mode": level3_mode,
        "status": "applied",
        "seed": level3_seed,
        "variant_index": materialized["variant_index"],
        "rewritten_files": materialized["rewritten_files"],
        "file_orders": materialized["file_orders"],
    }
    (output_dir / instance_id / "level3_runtime.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / instance_id / "level3_runtime.json").write_text(json.dumps(level3_info, indent=2))
    return {"level3": level3_info}


def maybe_apply_level4_semantic_rewrite(
    *,
    instance: dict,
    env: Environment,
    output_dir: Path,
    level4_mode: str,
    level4_verified_store: Path | None,
    level4_seed: int,
) -> dict:
    if level4_mode not in LEVEL4_VERIFIED_MODES:
        return {}
    if level4_verified_store is None:
        raise ValueError("Level 4 local semantic rewrite requires --level4-verified-store")
    if not hasattr(env, "container_id") or not hasattr(env, "config"):
        raise ValueError("Level 4 local semantic rewrite currently requires a docker-backed environment")

    instance_id = instance["instance_id"]
    record = load_verified_level4_record(level4_verified_store, instance_id, level4_mode)
    if not record:
        return {"level4": {"mode": level4_mode, "status": "skipped", "reason": "missing_verified_store_record"}}
    if record.get("status") == "skipped":
        return {"level4": {"mode": level4_mode, "status": "skipped", "reason": record.get("reason", "verified_store_skipped")}}

    variant = select_verified_level4_variant(record, instance_id=instance_id, level4_seed=level4_seed)
    if not variant:
        return {"level4": {"mode": level4_mode, "status": "skipped", "reason": "missing_verified_variant"}}

    mode_dir = Path(record["mode_dir"])
    variant_name = variant["name"]
    variant_dir = mode_dir / variant_name
    overlay_root = variant_dir / "level4_overlay"
    runtime_path = variant_dir / "level4_runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    tmp_root = output_dir / instance_id / "level4_overlay"
    for overlay_file in overlay_root.rglob("*"):
        if overlay_file.is_file():
            relative_path = overlay_file.relative_to(overlay_root).as_posix()
            copy_text_to_container(
                env.container_id,
                env.config.executable,
                relative_path,
                overlay_file.read_text(),
                tmp_root=tmp_root,
            )

    level4_info = {
        "mode": level4_mode,
        "status": runtime.get("status", "applied"),
        "seed": level4_seed,
        "variant_index": runtime.get("variant_index"),
        "runtime_seed": runtime.get("runtime_seed"),
        "variant_name": variant_name,
        "rewritten_files": runtime.get("rewritten_files", []),
        "candidate_ids": runtime.get("candidate_ids", []),
        "rewrite_kinds": runtime.get("rewrite_kinds", []),
        "verified_store": str(level4_verified_store),
    }
    for key in (
        "target_group",
        "target_group_size",
        "target_group_signature",
        "rewrite_summaries",
        "rewrite_strength",
        "generator_backend",
        "core_functions",
        "all_scope_functions",
        "all_changed_functions",
        "applied_rules",
        "core_changed",
    ):
        if key in runtime:
            level4_info[key] = runtime[key]
    for key in ("target_function", "target_path", "rewrite_summary"):
        if key in runtime:
            level4_info[key] = runtime[key]
    (output_dir / instance_id / "level4_runtime.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / instance_id / "level4_runtime.json").write_text(json.dumps(level4_info, indent=2))
    return {"level4": level4_info}


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    semantic_mappings: Path | None,
    semantic_seed: int,
    rename_repo: bool,
    enabled_layers: list[str] | None,
    translated_problems: dict | None = None,
    level3_mode: str = "",
    level3_trajectories: Path | None = None,
    level3_repo_root: Path | None = None,
    level3_verified_store: Path | None = None,
    level3_seed: int = 42,
    level3_submission_mode: str = "keep",
    level4_mode: str = "",
    level4_repo_root: Path | None = None,
    level4_verified_store: Path | None = None,
    level4_seed: int = 42,
) -> None:
    """Process a single SWEBench instance with optional semantic mapping."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    shutil.rmtree(instance_dir / "level3_overlay", ignore_errors=True)
    shutil.rmtree(instance_dir / "level4_overlay", ignore_errors=True)
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status = None
    result = None
    extra_info = {}
    project_name = instance_id.split("__", 1)[0]
    mapping_bundle_dir = None
    selected_mapping_bundle_dir = None
    selected_mapping_variant = None

    try:
        env = get_sb_environment(config, instance)
        extra_info.update(
            maybe_apply_level3_intra_file(
                instance=instance,
                env=env,
                output_dir=output_dir,
                level3_mode=level3_mode,
                level3_trajectories=level3_trajectories,
                level3_repo_root=level3_repo_root,
                level3_verified_store=level3_verified_store,
                level3_seed=level3_seed,
            )
        )
        extra_info.update(
            maybe_apply_level4_semantic_rewrite(
                instance=instance,
                env=env,
                output_dir=output_dir,
                level4_mode=level4_mode,
                level4_verified_store=level4_verified_store,
                level4_seed=level4_seed,
            )
        )

        # Set up semantic mapping if provided
        mapping_manager = None
        if semantic_mappings:
            mapping_bundle_dir = resolve_mapping_bundle_dir(semantic_mappings, instance_id)
            if mapping_bundle_dir.exists():
                selected_mapping_bundle_dir, selected_mapping_variant = select_mapping_bundle_dir(
                    mapping_bundle_dir,
                    semantic_seed=semantic_seed,
                )
                mapping_manager = SemanticMappingManager(
                    selected_mapping_bundle_dir,
                    seed=semantic_seed,
                    enabled_layers=enabled_layers,
                    project_name=project_name,
                )
                logger.info(f"Loaded semantic mapping for {instance_id}: {len(mapping_manager.forward_map)} identifiers")

                # Optionally rename repo directory
                if rename_repo:
                    # Use the mapping to find virtual name for the project
                    virtual_project = mapping_manager.forward_map.get(project_name)
                    if virtual_project:
                        progress_manager.update_instance_status(instance_id, "Renaming repo")
                        rename_repo_in_container(env, project_name, virtual_project)
            else:
                logger.warning(f"No mapping found for {instance_id} at {mapping_bundle_dir}")

        if translated_problems and instance_id in translated_problems:
            rec = translated_problems[instance_id]
            if isinstance(rec, dict) and rec.get("translated"):
                task = rec["translated"]
                logger.info(f"Using translated problem statement for {instance_id}")
            else:
                task = instance["problem_statement"]
        else:
            task = instance["problem_statement"]

        if mapping_manager is not None:
            task = mapping_manager.to_virtual_text(task)

        agent = MappedProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            mapping_manager=mapping_manager,
            **config.get("agent", {}),
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        raw_submission = info.get("submission")
        state_submissions = maybe_build_state_based_submissions(
            instance_id=instance_id,
            env=env,
            output_dir=output_dir,
            level3_mode=level3_mode,
            level3_repo_root=level3_repo_root,
            level3_info=extra_info.get("level3"),
            level3_submission_mode=level3_submission_mode,
            level4_mode=level4_mode,
            level4_repo_root=level4_repo_root,
            level4_info=extra_info.get("level4"),
        )
        if state_submissions and state_submissions.get("final_submission", "").strip():
            result = maybe_remap_submission(
                state_submissions["final_submission"],
                instance_id=instance_id,
                env=env,
                output_dir=output_dir,
                mapping_manager=mapping_manager,
                level3_mode=level3_mode,
                level3_repo_root=level3_repo_root,
                level3_info=extra_info.get("level3"),
                level3_submission_mode=level3_submission_mode,
            )
            extra_info["agent_submission"] = state_submissions.get("agent_submission", "")
            extra_info["raw_submission"] = raw_submission
        else:
            result = maybe_remap_submission(
                raw_submission,
                instance_id=instance_id,
                env=env,
                output_dir=output_dir,
                mapping_manager=mapping_manager,
                level3_mode=level3_mode,
                level3_repo_root=level3_repo_root,
                level3_info=extra_info.get("level3"),
                level3_submission_mode=level3_submission_mode,
            )
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                        **build_mapping_info(
                            mapping_manager=mapping_manager,
                            bundle_dir=selected_mapping_bundle_dir or mapping_bundle_dir,
                            bundle_root_dir=mapping_bundle_dir,
                            enabled_layers=enabled_layers,
                            semantic_seed=semantic_seed,
                            project_name=project_name,
                            bundle_variant=selected_mapping_variant,
                        ),
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset"),
    split: str = typer.Option("dev", "--split"),
    slice_spec: str = typer.Option("", "--slice"),
    filter_spec: str = typer.Option("", "--filter"),
    filter_file: str = typer.Option("", "--filter-file", help="File containing instance IDs to include, one per line"),
    shuffle: bool = typer.Option(False, "--shuffle"),
    output: str = typer.Option("", "-o", "--output"),
    workers: int = typer.Option(1, "-w", "--workers"),
    model: str | None = typer.Option(None, "-m", "--model"),
    model_class: str | None = typer.Option(None, "--model-class"),
    redo_existing: bool = typer.Option(False, "--redo-existing"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config"),
    environment_class: str | None = typer.Option(None, "--environment-class"),
    semantic_mappings: str = typer.Option("", "--semantic-mappings", help="Path to semantic mappings directory"),
    semantic_seed: int = typer.Option(42, "--semantic-seed", help="Seed for semantic mapping selection"),
    enabled_layers: list[str] | None = typer.Option(None, "--enabled-layer", help="Enable mapping layer, repeatable"),
    rename_repo: bool = typer.Option(False, "--rename-repo", help="Rename repo directory in container"),
    translated_problems: str = typer.Option(
        "",
        "--translated-problems",
        help="JSON file: instance_id -> {translated: ...} (e.g. re_issues/llm_django.json)",
    ),
    level3_mode: str = typer.Option("", "--level3-mode", help="Level 3 mode (currently: intra_file_reorder)"),
    level3_trajectories: str = typer.Option("", "--level3-trajectories", help="Path to baseline trajectory directory for Level 3"),
    level3_repo_root: str = typer.Option("", "--level3-repo-root", help="Path to local instance repo cache root for Level 3"),
    level3_verified_store: str = typer.Option("", "--level3-verified-store", help="Path to verified Level 3 perturbation store"),
    level3_seed: int = typer.Option(42, "--level3-seed", help="Seed for Level 3 variant selection"),
    level3_submission_mode: str = typer.Option("keep", "--level3-submission-mode", help="Level 3 submission mode: keep or remap"),
    level4_mode: str = typer.Option("", "--level4-mode", help="Level 4 mode (currently: llm_function_body_rewrite, agentic_function_body_rewrite)"),
    level4_repo_root: str = typer.Option("", "--level4-repo-root", help="Path to local instance repo cache root for Level 4"),
    level4_verified_store: str = typer.Option("", "--level4-verified-store", help="Path to verified Level 4 perturbation store"),
    level4_seed: int = typer.Option(42, "--level4-seed", help="Seed for Level 4 variant selection"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    apply_hf_offline_env_for_datasets()
    instances = load_swebench_instances(dataset_path, split)

    # Apply filter-file if provided
    if filter_file:
        filter_ids = set(Path(filter_file).read_text().strip().splitlines())
        instances = [i for i in instances if i["instance_id"] in filter_ids]
        logger.info(f"Filtered to {len(instances)} instances from filter file")

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [i for i in instances if i["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    semantic_mappings_path = Path(semantic_mappings) if semantic_mappings else None
    level3_trajectories_path = Path(level3_trajectories) if level3_trajectories else None
    level3_repo_root_path = Path(level3_repo_root) if level3_repo_root else None
    level3_verified_store_path = Path(level3_verified_store) if level3_verified_store else None
    level4_repo_root_path = Path(level4_repo_root) if level4_repo_root else level3_repo_root_path
    level4_verified_store_path = Path(level4_verified_store) if level4_verified_store else None

    translated_problems_data: dict | None = None
    if translated_problems:
        tp_path = Path(translated_problems)
        translated_problems_data = json.loads(tp_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded %s translated problem entries from %s",
            len(translated_problems_data),
            tp_path,
        )

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                    semantic_mappings_path,
                    semantic_seed,
                    rename_repo,
                    enabled_layers,
                    translated_problems_data,
                    level3_mode,
                    level3_trajectories_path,
                    level3_repo_root_path,
                    level3_verified_store_path,
                    level3_seed,
                    level3_submission_mode,
                    level4_mode,
                    level4_repo_root_path,
                    level4_verified_store_path,
                    level4_seed,
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
