import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import importlib.util
import os
from pathlib import Path

from glasses.runtime_alias import write_runtime_alias_bundle
from glasses.swebench_verified import parse_instance_test_lists
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name


_PATCH_PATH_RE = re.compile(r"^\+\+\+ b/([^\s]+)", re.MULTILINE)


def extract_changed_test_paths(test_patch: str) -> list[str]:
    paths = []
    for path in _PATCH_PATH_RE.findall(test_patch or ""):
        if path.startswith(("tests/", "test/")) or "/tests/" in path or "/test/" in path:
            paths.append(path)
    return paths


def build_instance_test_command(instance: dict) -> str:
    repo = instance.get("repo", "")
    test_paths = extract_changed_test_paths(instance.get("test_patch", ""))

    if repo == "django/django":
        labels = sorted({Path(path).parts[1] for path in test_paths if len(Path(path).parts) > 1})
        if labels:
            return "python tests/runtests.py " + " ".join(labels) + " --verbosity 0 --parallel=1"
        return "python tests/runtests.py --verbosity 0 --parallel=1"

    if test_paths:
        return "python -m pytest " + " ".join(test_paths) + " -q"

    tests = parse_instance_test_lists(instance)
    if tests["fail_to_pass"]:
        return "python -m pytest -q"
    raise ValueError(f"Could not derive test command for {instance.get('instance_id')}")


def build_module_path_aliases(layout_map: dict, *, repo_root_in_container: str = "/testbed") -> dict[str, str]:
    aliases = {}
    for source, target in (layout_map.get("path_map") or {}).items():
        aliases[source[:-3].replace("/", ".")] = f"{repo_root_in_container}/{target}"
    return aliases


def build_overlay_paths_from_repo(repo_root: Path, relative_paths: list[str]) -> dict[str, Path]:
    repo_root = Path(repo_root)
    return {path: repo_root / path for path in relative_paths}


def build_official_eval_script(test_spec, *, runtime_alias_prefix: str = "") -> str:
    if not runtime_alias_prefix:
        return test_spec.eval_script

    lines = test_spec.eval_script.splitlines()
    if len(lines) < 2:
        return test_spec.eval_script
    return "\n".join(lines[:2] + [runtime_alias_prefix] + lines[2:]) + "\n"


def _load_official_python_constants():
    module_path = os.environ.get(
        "SWEBENCH_PYTHON_CONSTANTS",
        str(Path(__file__).resolve().parents[1] / "SWE-bench" / "swebench" / "harness" / "constants" / "python.py"),
    )
    if not Path(module_path).exists():
        raise FileNotFoundError(
            "SWE-bench constants not found. Set SWEBENCH_PYTHON_CONSTANTS to "
            "swebench/harness/constants/python.py."
        )
    spec = importlib.util.spec_from_file_location("swebench_python_constants_local", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def get_official_python_specs(instance: dict) -> dict:
    module = _load_official_python_constants()
    return module.MAP_REPO_VERSION_TO_SPECS_PY[instance["repo"]][instance["version"]]


def get_official_python_test_directives(instance: dict) -> list[str]:
    directives = []
    for path in extract_changed_test_paths(instance.get("test_patch", "")):
        if any(path.endswith(ext) for ext in [".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml"]):
            continue
        directives.append(path)

    if instance.get("repo") == "django/django":
        transformed = []
        for path in directives:
            path = path[:-3] if path.endswith(".py") else path
            path = path[len("tests/") :] if path.startswith("tests/") else path
            transformed.append(path.replace("/", "."))
        return transformed
    return directives


def build_official_python_eval_script(instance: dict, *, runtime_alias_prefix: str = "") -> str:
    specs = get_official_python_specs(instance)
    test_cmd = " ".join([specs["test_cmd"], *get_official_python_test_directives(instance)]).strip()
    repo_directory = "/testbed"
    base_commit = instance["base_commit"]
    test_files = extract_changed_test_paths(instance.get("test_patch", ""))
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}" if test_files else "true"
    heredoc = "EOF_114329324912"
    apply_test_patch_command = (
        f"git apply -v - <<'{heredoc}'\n{instance['test_patch']}\n{heredoc}"
        if instance.get("test_patch")
        else "true"
    )
    commands = [
        "#!/bin/bash",
        "set -uxo pipefail",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        f"cd {repo_directory}",
    ]
    if runtime_alias_prefix:
        commands.append(runtime_alias_prefix)
    if "eval_commands" in specs:
        commands.extend(specs["eval_commands"])
    commands.extend(
        [
            f"git config --global --add safe.directory {repo_directory}",
            f"cd {repo_directory}",
            "git status",
            "git show",
            f"git -c core.fileMode=false diff {base_commit}",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            reset_tests_command,
            apply_test_patch_command,
            ": '>>>>> Start Test Output'",
            test_cmd,
            ": '>>>>> End Test Output'",
            reset_tests_command,
        ]
    )
    return "\n".join(commands) + "\n"


def _docker(*args: str, executable: str = "docker", timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [executable, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def classify_verification_result(result: dict) -> dict:
    status = result.get("status")
    if status != "test_failed":
        return result

    output = result.get("output", "") or ""
    test_failure_markers = (
        "AssertionError:",
        "FAIL:",
        "FAILED",
        "FAILED (failures=",
        "FAILED (errors=",
    )
    if any(marker in output for marker in test_failure_markers):
        classified = dict(result)
        classified["status"] = "runtime_ok_tests_failed"
        classified["failure_kind"] = "test_assertion_failure"
        return classified

    runtime_markers = (
        "ImportError:",
        "ModuleNotFoundError:",
        "cannot import name",
        "SyntaxError:",
        "Traceback (most recent call last):",
    )
    if any(marker in output for marker in runtime_markers):
        classified = dict(result)
        classified["status"] = "runtime_broken"
        classified["failure_kind"] = "import_error" if "import" in output.lower() else "runtime_error"
        return classified

    classified = dict(result)
    classified["status"] = "runtime_unknown_failure"
    classified["failure_kind"] = "unknown"
    return classified


def classify_official_eval_output(output: str, *, returncode: int | None) -> dict:
    if ">>>>> Patch Apply Failed" in output or ">>>>> Reset Failed" in output:
        return {
            "status": "runtime_broken",
            "failure_kind": "patch_or_reset_failure",
            "returncode": returncode,
            "output": output,
        }
    if ">>>>> Tests Errored" in output or ">>>>> Tests Timed Out" in output:
        return {
            "status": "runtime_broken",
            "failure_kind": "runtime_error",
            "returncode": returncode,
            "output": output,
        }
    if ">>>>> Start Test Output" not in output or ">>>>> End Test Output" not in output:
        return {
            "status": "runtime_unknown_failure",
            "failure_kind": "missing_test_markers",
            "returncode": returncode,
            "output": output,
        }
    test_output = output.split(">>>>> Start Test Output", 1)[1].split(">>>>> End Test Output", 1)[0]
    return classify_verification_result(
        {
            "status": "test_failed"
            if any(marker in test_output for marker in ("FAIL:", "FAILED", "ERROR", "Traceback (most recent call last):", "AssertionError:"))
            else "ok",
            "returncode": returncode,
            "output": test_output,
        }
    )


def verify_repo_in_instance_container(
    instance: dict,
    repo_root: Path,
    *,
    test_command: str | None = None,
    module_aliases: dict[str, str] | None = None,
    module_path_aliases: dict[str, str] | None = None,
    executable: str = "docker",
) -> dict:
    image = get_swebench_docker_image_name(instance)
    container_name = f"glasses-l3-{uuid.uuid4().hex[:10]}"
    test_command = test_command or build_instance_test_command(instance)
    repo_root = Path(repo_root)
    temp_patch = None
    alias_bundle = None

    start = _docker("run", "-d", "--name", container_name, "-w", "/testbed", image, "sleep", "2h", executable=executable)
    if start.returncode != 0:
        return {"status": "docker_start_failed", "image": image, "output": start.stdout}

    try:
        clear = _docker("exec", container_name, "bash", "-lc", "find /testbed -mindepth 1 -maxdepth 1 -exec rm -rf {} +", executable=executable)
        if clear.returncode != 0:
            return {"status": "container_prepare_failed", "image": image, "output": clear.stdout}

        copy_repo = _docker("cp", str(repo_root) + "/.", f"{container_name}:/testbed", executable=executable)
        if copy_repo.returncode != 0:
            return {"status": "repo_copy_failed", "image": image, "output": copy_repo.stdout}

        if module_aliases or module_path_aliases:
            alias_bundle = Path(tempfile.mkdtemp()) / "shim"
            write_runtime_alias_bundle(alias_bundle, module_aliases or {}, path_aliases=module_path_aliases or {})
            copy_alias = _docker("cp", str(alias_bundle) + "/.", f"{container_name}:/opt/glasses_runtime_shim", executable=executable)
            if copy_alias.returncode != 0:
                return {"status": "runtime_alias_copy_failed", "image": image, "output": copy_alias.stdout}

        if instance.get("test_patch"):
            with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as handle:
                handle.write(instance["test_patch"])
                temp_patch = Path(handle.name)
            copy_patch = _docker("cp", str(temp_patch), f"{container_name}:/tmp/test_patch.diff", executable=executable)
            if copy_patch.returncode != 0:
                return {"status": "test_patch_copy_failed", "image": image, "output": copy_patch.stdout}
            apply_patch = _docker(
                "exec",
                container_name,
                "bash",
                "-lc",
                "cd /testbed && git apply /tmp/test_patch.diff",
                executable=executable,
            )
            if apply_patch.returncode != 0:
                return {"status": "test_patch_apply_failed", "image": image, "output": apply_patch.stdout}

        prefix = "export PYTHONPATH=/opt/glasses_runtime_shim:/testbed:${PYTHONPATH}; " if (module_aliases or module_path_aliases) else ""
        run = _docker("exec", container_name, "bash", "-lc", f"{prefix}cd /testbed && {test_command}", executable=executable)
        return classify_verification_result({
            "status": "ok" if run.returncode == 0 else "test_failed",
            "image": image,
            "test_command": test_command,
            "returncode": run.returncode,
            "output": run.stdout,
        })
    finally:
        _docker("rm", "-f", container_name, executable=executable)
        if temp_patch and temp_patch.exists():
            temp_patch.unlink()
        if alias_bundle and alias_bundle.parent.exists():
            shutil.rmtree(alias_bundle.parent, ignore_errors=True)


def verify_repo_in_instance_container_official(
    instance: dict,
    repo_root: Path,
    *,
    overlay_paths: list[str],
    module_aliases: dict[str, str] | None = None,
    module_path_aliases: dict[str, str] | None = None,
    timeout: int | None = 600,
    namespace: str = "docker.io/swebench",
    tmp_root: Path | None = None,
) -> dict:
    repo_root = Path(repo_root)
    tmp_root = Path(tmp_root or Path.cwd() / "tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    image = get_swebench_docker_image_name(instance)
    run_id = f"glasses-{uuid.uuid4().hex[:10]}"
    log_dir = tmp_root / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    container_name = f"glasses-official-{uuid.uuid4().hex[:10]}"
    alias_bundle = None

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
        return {"status": "docker_start_failed", "image": image, "output": start.stdout}

    try:
        overlay_map = build_overlay_paths_from_repo(repo_root, overlay_paths)
        for rel_path, full_path in overlay_map.items():
            if not full_path.exists():
                return {
                    "status": "overlay_missing",
                    "instance_id": instance["instance_id"],
                    "missing_path": rel_path,
                }
            copy_result = _docker("cp", str(full_path), f"{container_name}:/testbed/{rel_path}")
            if copy_result.returncode != 0:
                return {"status": "overlay_copy_failed", "image": image, "output": copy_result.stdout}

        runtime_alias_prefix = ""
        if module_aliases or module_path_aliases:
            alias_bundle = Path(tempfile.mkdtemp(dir=tmp_root)) / "shim"
            write_runtime_alias_bundle(alias_bundle, module_aliases or {}, path_aliases=module_path_aliases or {})
            for shim_file in alias_bundle.iterdir():
                copy_result = _docker("cp", str(shim_file), f"{container_name}:/opt/glasses_runtime_shim/{shim_file.name}")
                if copy_result.returncode != 0:
                    return {"status": "runtime_alias_copy_failed", "image": image, "output": copy_result.stdout}
            runtime_alias_prefix = "export PYTHONPATH=/opt/glasses_runtime_shim:/testbed:${PYTHONPATH}"

        eval_file = log_dir / "eval.sh"
        eval_file.write_text(build_official_python_eval_script(instance, runtime_alias_prefix=runtime_alias_prefix))
        copy_eval = _docker("cp", str(eval_file), f"{container_name}:/eval.sh")
        if copy_eval.returncode != 0:
            return {"status": "eval_script_copy_failed", "image": image, "output": copy_eval.stdout}

        run = _docker("exec", container_name, "bash", "-lc", "/bin/bash /eval.sh")
        return classify_official_eval_output(run.stdout, returncode=run.returncode) | {"image": image}
    finally:
        _docker("rm", "-f", container_name)
        if alias_bundle and alias_bundle.parent.exists():
            shutil.rmtree(alias_bundle.parent, ignore_errors=True)
