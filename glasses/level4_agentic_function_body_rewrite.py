import json
import re
import shutil
import subprocess
from pathlib import Path

from glasses.swebench_verified import parse_instance_test_lists
from minisweagent.agents import get_agent
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.utils.serialize import UNSET, recursive_merge

MODE = "agentic_function_body_rewrite"
FULL_TARGET_GROUP_SCOPE = "full_target_group"
GOLDEN_PATCH_LOCAL_SCOPE = "golden_patch_local"


def build_agentic_rewrite_task(
    *,
    instance_id: str,
    target_group: list[dict],
    target_sources: dict[str, dict],
    pass_to_pass_tests: list[str],
    rewrite_scope: str = FULL_TARGET_GROUP_SCOPE,
    golden_patch_local_context: dict[str, str] | None = None,
    previous_attempt_feedback: dict | None = None,
) -> str:
    payload = {
        "instance_id": instance_id,
        "target_group": target_group,
        "target_sources": target_sources,
        "pass_to_pass_tests": pass_to_pass_tests,
        "rewrite_scope": rewrite_scope,
    }
    if golden_patch_local_context:
        payload["golden_patch_local_context"] = golden_patch_local_context
    if previous_attempt_feedback:
        payload["previous_attempt_feedback"] = previous_attempt_feedback
    if rewrite_scope == GOLDEN_PATCH_LOCAL_SCOPE:
        instructions = (
            "You are operating inside the benchmark Docker environment. "
            "Your job is to produce a local, conservative, behavior-preserving patch, not a broad refactor. "
            "First locate the target code, inspect the listed PASS_TO_PASS tests, and understand the relevant surrounding context. "
            "Then rewrite only the code closely related to the golden patch neighborhood. "
            "Prefer the smallest viable local rewrite near the golden patch hunk. "
            "If a stronger local rewrite is too risky, fall back to safer local transformations such as introducing local temporary variables, "
            "renaming local variables, and reordering independent local statements when dependency and side-effect order are preserved. "
            "Do not refactor unrelated parts of the target functions. "
            "Do not change function signatures, decorators, imports, global state, exception types, or control flow outside the local patch neighborhood. "
            "Do not introduce unrelated helpers or move logic across functions unless that is strictly necessary for a local equivalent rewrite. "
            "Preserve behavior exactly, especially for the listed PASS_TO_PASS tests. "
            "If previous_attempt_feedback is present, use it directly: avoid repeating the same patch shape and avoid reintroducing the same failing behavior. "
            "Before submitting, ensure the patch is a valid unified diff that can be applied cleanly with git apply. "
            "Submit only the rewrite patch.\n\n"
        )
    else:
        instructions = (
            "You are operating inside the benchmark Docker environment. "
            "First locate the target code, then understand the relevant surrounding context, "
            "then rewrite the target_group functions so the implementation looks significantly different "
            "while behavior stays equivalent. Work around the listed PASS_TO_PASS tests and keep them passing. "
            "Submit only the rewrite patch.\n\n"
        )
    return instructions + f"{json.dumps(payload, ensure_ascii=False, indent=2)}"


def collect_agentic_changed_files(repo_root: Path, changed_paths: list[str]) -> dict[str, str]:
    changed = {}
    for rel in sorted(set(changed_paths)):
        if not rel.endswith(".py"):
            continue
        path = repo_root / rel
        if path.exists():
            changed[rel] = path.read_text()
    return changed


def prepare_agentic_repo_copy(repo_root: Path, work_dir: Path) -> Path:
    agent_repo_root = work_dir / "repo"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(repo_root, agent_repo_root, dirs_exist_ok=True)
    return agent_repo_root


def run_mini_agentic_rewrite(
    *,
    instance: dict,
    repo_root: Path,
    task: str,
    model_name: str,
    output_path: Path,
    config_path: Path,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = recursive_merge(
        get_config_from_spec(str(config_path)),
        {
            "run": {"task": task},
            "model": {"model_name": model_name, "model_kwargs": {"temperature": 1}},
            "agent": {"mode": "yolo", "confirm_exit": False, "output_path": output_path},
        },
    )
    if config.get("run", {}).get("task", UNSET) is UNSET:
        raise ValueError("agentic rewrite task missing")
    model = get_model(config=config.get("model", {}))
    env = get_sb_environment(config, instance)
    agent = get_agent(model, env, config.get("agent", {}), default_type="default")
    info = agent.run(config["run"]["task"])
    return {
        "submission": info.get("submission", ""),
        "trajectory_path": str(output_path),
    }


def apply_agentic_submission_patch(repo_root: Path, submission: str, patch_path: Path) -> None:
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(submission)
    try:
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        (patch_path.parent / "git_apply_stdout.txt").write_text(exc.stdout or "")
        (patch_path.parent / "git_apply_stderr.txt").write_text(exc.stderr or "")
        raise


def extract_agentic_changed_paths(submission: str) -> list[str]:
    changed = []
    for line in submission.splitlines():
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if not match:
            continue
        _old_path, new_path = match.groups()
        if new_path != "/dev/null":
            changed.append(new_path)
    return changed


def get_agentic_pass_to_pass_tests(instance: dict) -> list[str]:
    return parse_instance_test_lists(instance).get("pass_to_pass", [])
