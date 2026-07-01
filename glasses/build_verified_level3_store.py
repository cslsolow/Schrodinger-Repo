#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import subprocess
import shutil
import sys
import time
import types
import uuid
from pathlib import Path

from datasets import Dataset

try:
    from glasses.verified_dataset_paths import default_verified_arrow
except ModuleNotFoundError:
    from verified_dataset_paths import default_verified_arrow

ROOT = Path(__file__).resolve().parents[1]
SWEBENCH_ROOT = Path(os.environ.get("SWEBENCH_ROOT", ROOT / "SWE-bench"))
sys.path.insert(0, str(ROOT / "src"))

try:
    from glasses.docker_verifier import _docker, build_official_python_eval_script
    from glasses.intra_file_reorder import build_intra_file_reorder_plan, materialize_intra_file_variant
except ModuleNotFoundError:
    from docker_verifier import _docker, build_official_python_eval_script
    from intra_file_reorder import build_intra_file_reorder_plan, materialize_intra_file_variant
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name
from minisweagent.run.benchmarks.swebench_mapped import build_git_patch_from_bytes


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_harness_modules():
    swebench_pkg = sys.modules.setdefault("swebench", types.ModuleType("swebench"))
    swebench_pkg.__path__ = [str(SWEBENCH_ROOT / "swebench")]
    harness_pkg = sys.modules.setdefault("swebench.harness", types.ModuleType("swebench.harness"))
    harness_pkg.__path__ = [str(SWEBENCH_ROOT / "swebench" / "harness")]
    test_spec_pkg = sys.modules.setdefault("swebench.harness.test_spec", types.ModuleType("swebench.harness.test_spec"))
    test_spec_pkg.__path__ = [str(SWEBENCH_ROOT / "swebench" / "harness" / "test_spec")]

    load_module("swebench.harness.constants", SWEBENCH_ROOT / "swebench" / "harness" / "constants" / "__init__.py")
    load_module("swebench.harness.dockerfiles", SWEBENCH_ROOT / "swebench" / "harness" / "dockerfiles" / "__init__.py")
    load_module("swebench.harness.test_spec.javascript", SWEBENCH_ROOT / "swebench" / "harness" / "test_spec" / "javascript.py")
    load_module("swebench.harness.test_spec.python", SWEBENCH_ROOT / "swebench" / "harness" / "test_spec" / "python.py")
    load_module("swebench.harness.test_spec.create_scripts", SWEBENCH_ROOT / "swebench" / "harness" / "test_spec" / "create_scripts.py")
    test_spec_mod = load_module("swebench.harness.test_spec.test_spec", SWEBENCH_ROOT / "swebench" / "harness" / "test_spec" / "test_spec.py")
    load_module("swebench.harness.log_parsers", SWEBENCH_ROOT / "swebench" / "harness" / "log_parsers" / "__init__.py")
    grading_mod = load_module("swebench.harness.grading", SWEBENCH_ROOT / "swebench" / "harness" / "grading.py")
    constants_mod = sys.modules["swebench.harness.constants"]
    return constants_mod, grading_mod, test_spec_mod


CONSTANTS, GRADING, TEST_SPEC = load_harness_modules()
KEY_INSTANCE_ID = CONSTANTS.KEY_INSTANCE_ID
KEY_MODEL = CONSTANTS.KEY_MODEL
KEY_PREDICTION = CONSTANTS.KEY_PREDICTION
get_eval_report = GRADING.get_eval_report
make_test_spec = TEST_SPEC.make_test_spec

DEFAULT_ARROW = default_verified_arrow()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter-file",
        default=str(ROOT / "instances_django_mapped_merged_minus_instances_django_baseline_gpt54mini_resolved.txt"),
    )
    parser.add_argument(
        "--baseline-trajectories",
        default=str(ROOT / ".." / "v2_baseline" / "mini-swe-agent" / "trajectories_baseline_gpt54mini_django"),
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT / "repos"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "output" / "verified_perturbations" / "level3"),
    )
    parser.add_argument("--mode", default="intra_file_reorder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--variant-count", type=int, default=5)
    parser.add_argument("--target-variants", type=int, default=1)
    parser.add_argument("--preflight-timeout", type=int, default=1800)
    parser.add_argument("--instance-time-budget", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--tmp-root", default=str(ROOT / "tmp" / "verified_level3_builder"))
    return parser.parse_args()


def load_verified_dataset_map(instance_ids: list[str], arrow_path: Path) -> dict[str, dict]:
    dataset = Dataset.from_file(str(arrow_path))
    needed = set(instance_ids)
    result = {}
    for item in dataset:
        inst = item.get("instance_id")
        if inst in needed:
            result[inst] = dict(item)
            if len(result) == len(needed):
                break
    missing = [inst for inst in instance_ids if inst not in result]
    if missing:
        raise KeyError(f"Missing verified instances: {missing[:5]}")
    return result


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_overlay(overlay_root: Path, files: dict[str, str]) -> None:
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_root.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        target = overlay_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def variant_dir(mode_dir: Path, index: int) -> Path:
    return mode_dir / f"variant_{index}"


def ensure_variantized_mode_dir(mode_dir: Path) -> None:
    if not mode_dir.exists():
        return
    has_top_level = (mode_dir / "level3_runtime.json").exists() or (mode_dir / "p2p_status.json").exists()
    first_variant = variant_dir(mode_dir, 0)
    if not has_top_level or first_variant.exists():
        return
    first_variant.mkdir(parents=True, exist_ok=True)
    for name in ["level3_runtime.json", "p2p_status.json", "level3_patch.diff"]:
        src = mode_dir / name
        if src.exists():
            shutil.copy2(src, first_variant / name)
    src_overlay = mode_dir / "level3_overlay"
    dst_overlay = first_variant / "level3_overlay"
    if src_overlay.exists() and not dst_overlay.exists():
        dst_overlay.mkdir(parents=True, exist_ok=True)
        for item in src_overlay.rglob("*"):
            if not item.is_file():
                continue
            try:
                relative_path = item.relative_to(src_overlay)
                target = dst_overlay / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
            except FileNotFoundError:
                continue


def list_existing_variants(mode_dir: Path) -> list[Path]:
    ensure_variantized_mode_dir(mode_dir)
    return sorted(
        [p for p in mode_dir.glob("variant_*") if p.is_dir()],
        key=lambda path: int(path.name.split("_", 1)[1]),
    )


def sync_default_variant(mode_dir: Path) -> None:
    variants = list_existing_variants(mode_dir)
    if not variants:
        return
    first = variants[0]
    for name in ["level3_runtime.json", "p2p_status.json", "level3_patch.diff"]:
        src = first / name
        dst = mode_dir / name
        if src.exists():
            shutil.copy2(src, dst)
        elif dst.exists():
            dst.unlink()
    src_overlay = first / "level3_overlay"
    dst_overlay = mode_dir / "level3_overlay"
    if dst_overlay.exists():
        shutil.rmtree(dst_overlay)
    if src_overlay.exists():
        shutil.copytree(src_overlay, dst_overlay)


def load_existing_variant_records(mode_dir: Path) -> list[dict]:
    records = []
    for variant_path in list_existing_variants(mode_dir):
        runtime_path = variant_path / "level3_runtime.json"
        status_path = variant_path / "p2p_status.json"
        if not runtime_path.exists() or not status_path.exists():
            continue
        runtime = json.loads(runtime_path.read_text())
        status = json.loads(status_path.read_text())
        file_orders = runtime.get("file_orders", {})
        signature = json.dumps(
            {
                "rewritten_files": runtime.get("rewritten_files", []),
                "file_orders": file_orders,
            },
            sort_keys=True,
        )
        records.append(
            {
                "name": variant_path.name,
                "path": variant_path,
                "runtime": runtime,
                "status": status,
                "signature": signature,
            }
        )
    return records


def ensure_repo_cache(instance: dict, repo_root: Path) -> Path:
    repo_root = Path(repo_root)
    if repo_root.exists():
        return repo_root

    repo_root.parent.mkdir(parents=True, exist_ok=True)
    container_name = f"glasses-repo-cache-{uuid.uuid4().hex[:10]}"
    image = get_swebench_docker_image_name(instance)
    start = _docker(
        "run",
        "-d",
        "--name",
        container_name,
        "-w",
        "/testbed",
        image,
        "tail",
        "-f",
        "/dev/null",
    )
    if start.returncode != 0:
        raise RuntimeError(f"failed to start cache container: {start.stdout}")

    try:
        copy_result = _docker("cp", f"{container_name}:/testbed", str(repo_root.parent))
        if copy_result.returncode != 0:
            raise RuntimeError(f"failed to copy repo cache: {copy_result.stdout}")
        copied = repo_root.parent / "testbed"
        if copied != repo_root:
            copied.rename(repo_root)
    finally:
        _docker("rm", "-f", container_name)

    return repo_root


def build_l3_patch(repo_root: Path, files: dict[str, str], tmp_root: Path) -> str:
    patch_parts = []
    for relative_path, content in files.items():
        base_path = repo_root / relative_path
        base_bytes = base_path.read_bytes() if base_path.exists() else None
        patch = build_git_patch_from_bytes(
            base_bytes,
            content.encode("utf-8", errors="surrogateescape"),
            relative_path,
            tmp_root=tmp_root / "patch",
        )
        if patch:
            patch_parts.append(patch)
    return "".join(patch_parts)


def restrict_plan(plan: dict, keep_paths: list[str]) -> dict:
    keep = set(keep_paths)
    files = [item for item in plan.get("files", []) if item["path"] in keep]
    variants = []
    for variant in plan.get("variants", []):
        file_orders = {path: orders for path, orders in variant["file_orders"].items() if path in keep}
        if file_orders:
            variants.append(
                {
                    "index": variant["index"],
                    "nonce": variant.get("nonce"),
                    "file_orders": file_orders,
                }
            )
    return {
        **plan,
        "files": files,
        "variants": variants,
    }


def run_official_preflight(
    instance: dict,
    overlay_root: Path,
    overlay_paths: list[str],
    l3_patch: str,
    log_dir: Path,
    timeout: int | None,
) -> dict:
    container_name = f"glasses-verified-{uuid.uuid4().hex[:10]}"
    image = get_swebench_docker_image_name(instance)
    start = _docker(
        "run",
        "-d",
        "--name",
        container_name,
        "-w",
        "/testbed",
        image,
        "tail",
        "-f",
        "/dev/null",
    )
    if start.returncode != 0:
        return {"status": "docker_start_failed", "output": start.stdout}

    try:
        for rel_path in overlay_paths:
            src = overlay_root / rel_path
            if not src.exists():
                return {"status": "overlay_missing", "path": rel_path}
            copy_result = _docker("cp", str(src), f"{container_name}:/testbed/{rel_path}")
            if copy_result.returncode != 0:
                return {"status": "overlay_copy_failed", "output": copy_result.stdout}

        eval_file = log_dir / "eval.sh"
        eval_file.write_text(build_official_python_eval_script(instance))
        copy_eval = _docker("cp", str(eval_file), f"{container_name}:/eval.sh")
        if copy_eval.returncode != 0:
            return {"status": "eval_script_copy_failed", "output": copy_eval.stdout}

        try:
            run = _docker("exec", container_name, "bash", "-lc", "/bin/bash /eval.sh", timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            (log_dir / "test_output.txt").write_text(output)
            return {"status": "eval_timeout", "returncode": None, "output": output, "timeout": timeout}
        test_output_path = log_dir / "test_output.txt"
        test_output_path.write_text(run.stdout)

        pred = {
            KEY_MODEL: "level3-preflight",
            KEY_INSTANCE_ID: instance["instance_id"],
            KEY_PREDICTION: l3_patch,
        }
        report = get_eval_report(
            make_test_spec(instance),
            pred,
            str(test_output_path),
            include_tests_status=True,
        )
        write_json(log_dir / "report.json", report)
        payload = report[instance["instance_id"]]
        return {
            "status": "ok" if payload["patch_successfully_applied"] else "runtime_broken",
            "report": payload,
            "returncode": run.returncode,
        }
    finally:
        _docker("rm", "-f", container_name)


def pass_to_pass_clean(result: dict) -> bool:
    report = result.get("report") or {}
    tests = report.get("tests_status") or {}
    p2p = tests.get("PASS_TO_PASS") or {}
    f2p = tests.get("FAIL_TO_PASS") or {}
    return (
        report.get("patch_successfully_applied")
        and not p2p.get("failure")
        and bool(f2p.get("failure"))
    )


def summarize_failure(result: dict) -> dict:
    report = result.get("report") or {}
    tests = report.get("tests_status") or {}
    return {
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "patch_successfully_applied": report.get("patch_successfully_applied"),
        "fail_to_pass_failure": (tests.get("FAIL_TO_PASS") or {}).get("failure", []),
        "pass_to_pass_failure": (tests.get("PASS_TO_PASS") or {}).get("failure", []),
    }


def collect_plan_variants(
    instance: dict,
    repo_root: Path,
    plan: dict,
    runtime_seed_base: int,
    work_dir: Path,
    *,
    want: int,
    preflight_timeout: int | None,
    seen_signatures: set[str],
) -> tuple[list[dict], dict]:
    chosen_variants = []
    attempt_log = []
    for variant_idx, _ in enumerate(plan.get("variants", [])):
        if len(chosen_variants) >= want:
            break
        runtime_seed = runtime_seed_base + variant_idx
        materialized = materialize_intra_file_variant(repo_root, plan, runtime_seed=runtime_seed)
        if materialized.get("status") != "ok" or not materialized.get("rewritten_files"):
            attempt_log.append(
                {
                    "variant_idx": variant_idx,
                    "runtime_seed": runtime_seed,
                    "status": materialized.get("status", "skipped"),
                    "rewritten_files": materialized.get("rewritten_files", []),
                }
            )
            continue

        signature = json.dumps(
            {
                "rewritten_files": materialized["rewritten_files"],
                "file_orders": materialized["file_orders"],
            },
            sort_keys=True,
        )
        if signature in seen_signatures:
            attempt_log.append(
                {
                    "variant_idx": variant_idx,
                    "runtime_seed": runtime_seed,
                    "rewritten_files": materialized["rewritten_files"],
                    "result": {"status": "duplicate_signature"},
                }
            )
            continue

        overlay_root = work_dir / "overlay"
        write_overlay(overlay_root, materialized["files"])
        l3_patch = build_l3_patch(repo_root, materialized["files"], work_dir / f"variant_{variant_idx}")
        log_dir = work_dir / "attempts" / f"variant_{variant_idx}"
        log_dir.mkdir(parents=True, exist_ok=True)
        result = run_official_preflight(
            instance,
            overlay_root,
            materialized["rewritten_files"],
            l3_patch,
            log_dir,
            preflight_timeout,
        )
        attempt_log.append(
            {
                "variant_idx": variant_idx,
                "runtime_seed": runtime_seed,
                "rewritten_files": materialized["rewritten_files"],
                "result": summarize_failure(result),
            }
        )
        if pass_to_pass_clean(result):
            chosen_variants.append(
                {
                    "runtime_seed": runtime_seed,
                    "variant_index": materialized["variant_index"],
                    "rewritten_files": materialized["rewritten_files"],
                    "file_orders": materialized["file_orders"],
                    "files": materialized["files"],
                    "l3_patch": l3_patch,
                    "report": result["report"],
                    "signature": signature,
                }
            )
            seen_signatures.add(signature)
    return chosen_variants, {"attempts": attempt_log}


def write_verified_variant(mode_dir: Path, slot_index: int, chosen: dict, active_files: list[str], diagnostics: list[dict], instance_id: str, seed: int) -> None:
    slot_dir = variant_dir(mode_dir, slot_index)
    overlay_root = slot_dir / "level3_overlay"
    write_overlay(overlay_root, chosen["files"])
    runtime = {
        "mode": "intra_file_reorder",
        "status": "applied",
        "seed": seed,
        "variant_index": chosen["variant_index"],
        "runtime_seed": chosen["runtime_seed"],
        "rewritten_files": chosen["rewritten_files"],
        "file_orders": chosen["file_orders"],
    }
    write_json(slot_dir / "level3_runtime.json", runtime)
    write_json(
        slot_dir / "p2p_status.json",
        {
            "instance_id": instance_id,
            "status": "verified",
            "active_files": active_files,
            "report": chosen["report"],
            "diagnostics": diagnostics,
        },
    )
    (slot_dir / "level3_patch.diff").write_text(chosen["l3_patch"])


def verify_instance(
    instance: dict,
    traj_path: Path,
    repo_root: Path,
    mode_dir: Path,
    tmp_root: Path,
    *,
    seed: int,
    top_k: int,
    variant_count: int,
    target_variants: int,
    preflight_timeout: int | None,
    instance_time_budget: int | None,
    redo: bool,
) -> dict:
    plan = build_intra_file_reorder_plan(traj_path, repo_root, top_k=top_k, variant_count=variant_count, seed=seed)
    existing = [] if redo else load_existing_variant_records(mode_dir)
    seen_signatures = {record["signature"] for record in existing}
    if plan["status"] == "no_candidate_files":
        runtime = {
            "mode": "intra_file_reorder",
            "status": "skipped",
            "reason": "no_candidate_files",
            "seed": seed,
            "variant_index": None,
            "rewritten_files": [],
            "file_orders": {},
        }
        if redo and mode_dir.exists():
            shutil.rmtree(mode_dir)
        mode_dir.mkdir(parents=True, exist_ok=True)
        write_json(mode_dir / "level3_runtime.json", runtime)
        write_json(mode_dir / "p2p_status.json", {"instance_id": instance["instance_id"], "status": "skipped", "reason": "no_candidate_files"})
        return {"instance_id": instance["instance_id"], "status": "skipped", "reason": "no_candidate_files"}

    ranked_paths = [item["path"] for item in sorted(plan["files"], key=lambda item: (-item["score"], item["path"]))]
    active = ranked_paths[:]
    diagnostics = []
    chosen_variants = []
    start_time = time.monotonic()

    def budget_exhausted() -> bool:
        return instance_time_budget is not None and (time.monotonic() - start_time) >= instance_time_budget

    if redo and mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)

    for record in existing:
        if record["status"].get("status") == "verified":
            chosen_variants.append(
                {
                    "runtime_seed": record["runtime"].get("runtime_seed"),
                    "variant_index": record["runtime"].get("variant_index"),
                    "rewritten_files": record["runtime"].get("rewritten_files", []),
                    "file_orders": record["runtime"].get("file_orders", {}),
                    "report": record["status"].get("report", {}),
                    "active_files": record["status"].get("active_files", []),
                    "name": record["name"],
                }
            )

    while active:
        if len(chosen_variants) >= target_variants:
            break
        if budget_exhausted():
            break
        restricted = restrict_plan(plan, active)
        work_dir = tmp_root / instance["instance_id"] / f"subset_{len(active)}"
        new_variants, meta = collect_plan_variants(
            instance,
            repo_root,
            restricted,
            seed,
            work_dir,
            want=target_variants - len(chosen_variants),
            preflight_timeout=preflight_timeout,
            seen_signatures=seen_signatures,
        )
        diagnostics.append({"active_files": active[:], **meta})
        if new_variants:
            for chosen in new_variants:
                slot_index = len(chosen_variants)
                write_verified_variant(mode_dir, slot_index, chosen, active, diagnostics, instance["instance_id"], seed)
                chosen_variants.append(
                    {
                        "runtime_seed": chosen["runtime_seed"],
                        "variant_index": chosen["variant_index"],
                        "rewritten_files": chosen["rewritten_files"],
                        "file_orders": chosen["file_orders"],
                        "report": chosen["report"],
                        "active_files": active[:],
                        "name": f"variant_{slot_index}",
                    }
                )
                if len(chosen_variants) >= target_variants:
                    break

        if len(active) == 1:
            break

        removed = None
        for candidate in sorted(active, key=lambda path: next(item["score"] for item in plan["files"] if item["path"] == path)):
            if len(chosen_variants) >= target_variants:
                break
            if budget_exhausted():
                break
            test_active = [path for path in active if path != candidate]
            restricted = restrict_plan(plan, test_active)
            work_dir = tmp_root / instance["instance_id"] / f"probe_drop_{candidate.replace('/', '_')}"
            new_variants, meta = collect_plan_variants(
                instance,
                repo_root,
                restricted,
                seed,
                work_dir,
                want=target_variants - len(chosen_variants),
                preflight_timeout=preflight_timeout,
                seen_signatures=seen_signatures,
            )
            diagnostics.append({"probe_drop": candidate, "active_files": test_active, **meta})
            if new_variants:
                for chosen in new_variants:
                    slot_index = len(chosen_variants)
                    write_verified_variant(mode_dir, slot_index, chosen, test_active, diagnostics, instance["instance_id"], seed)
                    chosen_variants.append(
                        {
                            "runtime_seed": chosen["runtime_seed"],
                            "variant_index": chosen["variant_index"],
                            "rewritten_files": chosen["rewritten_files"],
                            "file_orders": chosen["file_orders"],
                            "report": chosen["report"],
                            "active_files": test_active[:],
                            "name": f"variant_{slot_index}",
                        }
                    )
                    if len(chosen_variants) >= target_variants:
                        break
                active = test_active
                removed = candidate
                break
        if removed is None:
            active = active[:-1]

    if chosen_variants:
        sync_default_variant(mode_dir)
        return {
            "instance_id": instance["instance_id"],
            "status": "verified",
            "variants": [
                {
                    "name": item["name"],
                    "active_files": item["active_files"],
                    "runtime_seed": item["runtime_seed"],
                    "variant_index": item["variant_index"],
                }
                for item in chosen_variants
            ],
        }

    if budget_exhausted():
        runtime = {
            "mode": "intra_file_reorder",
            "status": "skipped",
            "reason": "instance_time_budget_exhausted",
            "seed": seed,
            "variant_index": None,
            "rewritten_files": [],
            "file_orders": {},
        }
        write_json(mode_dir / "level3_runtime.json", runtime)
        write_json(
            mode_dir / "p2p_status.json",
            {
                "instance_id": instance["instance_id"],
                "status": "skipped",
                "reason": "instance_time_budget_exhausted",
                "diagnostics": diagnostics,
            },
        )
        return {"instance_id": instance["instance_id"], "status": "skipped", "reason": "instance_time_budget_exhausted"}

    runtime = {
        "mode": "intra_file_reorder",
        "status": "skipped",
        "reason": "p2p_failed_all_subsets",
        "seed": seed,
        "variant_index": None,
        "rewritten_files": [],
        "file_orders": {},
    }
    write_json(mode_dir / "level3_runtime.json", runtime)
    write_json(
        mode_dir / "p2p_status.json",
        {
            "instance_id": instance["instance_id"],
            "status": "skipped",
            "reason": "p2p_failed_all_subsets",
            "diagnostics": diagnostics,
        },
    )
    return {"instance_id": instance["instance_id"], "status": "skipped", "reason": "p2p_failed_all_subsets"}


def main() -> None:
    args = parse_args()
    filter_ids = [line.strip() for line in Path(args.filter_file).read_text().splitlines() if line.strip()]
    if args.limit:
        filter_ids = filter_ids[: args.limit]

    baseline_root = Path(args.baseline_trajectories)
    repo_cache_root = Path(args.repo_root)
    output_root = Path(args.output_root)
    tmp_root = Path(args.tmp_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    instances = load_verified_dataset_map(filter_ids, DEFAULT_ARROW)
    index_path = output_root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    for instance_id in filter_ids:
        mode_dir = output_root / instance_id / args.mode
        if mode_dir.exists() and not args.redo:
            existing = load_existing_variant_records(mode_dir)
            if existing:
                sync_default_variant(mode_dir)
                verified_variants = [
                    {
                        "name": record["name"],
                        "active_files": record["status"].get("active_files", []),
                        "runtime_seed": record["runtime"].get("runtime_seed"),
                        "variant_index": record["runtime"].get("variant_index"),
                    }
                    for record in existing
                    if record["status"].get("status") == "verified"
                ]
                if len(verified_variants) >= args.target_variants:
                    index[instance_id] = {"instance_id": instance_id, "status": "verified", "variants": verified_variants[: args.target_variants]}
                    write_json(index_path, index)
                    continue
            status_file = mode_dir / "p2p_status.json"
            if status_file.exists():
                status_data = json.loads(status_file.read_text())
                if status_data.get("status") == "skipped":
                    index[instance_id] = status_data
                    write_json(index_path, index)
                    continue

        instance = instances[instance_id]
        traj_path = baseline_root / instance_id / f"{instance_id}.traj.json"
        if not traj_path.exists():
            status = {"instance_id": instance_id, "status": "skipped", "reason": "missing_baseline_trajectory"}
            write_json(mode_dir / "p2p_status.json", status)
            index[instance_id] = status
            write_json(index_path, index)
            print(instance_id, status["status"], status["reason"], flush=True)
            continue

        repo_root = ensure_repo_cache(instance, repo_cache_root / instance_id)

        status = verify_instance(
            instance,
            traj_path,
            repo_root,
            mode_dir,
            tmp_root,
            seed=args.seed,
            top_k=args.top_k,
            variant_count=args.variant_count,
            target_variants=args.target_variants,
            preflight_timeout=args.preflight_timeout,
            instance_time_budget=args.instance_time_budget,
            redo=args.redo,
        )
        index[instance_id] = status
        write_json(index_path, index)
        print(
            instance_id,
            status["status"],
            status.get("reason", len(status.get("variants", status.get("active_files", [])))),
            flush=True,
        )


if __name__ == "__main__":
    main()
