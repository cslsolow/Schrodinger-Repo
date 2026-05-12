import json
import os
from pathlib import Path
import subprocess
import sys


def _base_env(project_root):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root / 'src'}:{project_root}:{env.get('PYTHONPATH', '')}"
    return env


def test_level2_bundle_generation_writes_repo_bundle(tmp_path):
    repo_root = tmp_path / "repos"
    repo_dir = repo_root / "swe-bench_demo__demo-1"
    pkg = repo_dir / "demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "query.py").write_text(
        "class DemoThing:\n"
        "    pass\n\n"
        "def build_query():\n"
        "    return DemoThing()\n"
    )
    nested_pkg = pkg / "subpkg"
    nested_pkg.mkdir()
    (nested_pkg / "__init__.py").write_text("")
    (nested_pkg / "fetcher.py").write_text(
        "def fetch_payload():\n"
        "    return 1\n"
    )

    output_dir = tmp_path / "output"
    script_path = Path(__file__).resolve().parents[2] / "glasses" / "main.py"
    project_root = Path(__file__).resolve().parents[2]
    env = _base_env(project_root)

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--instance-id",
            "demo__demo-1",
            "--seed",
            "42",
            "--mapping-mode",
            "namespace_l2",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=project_root,
    )

    bundle_root = output_dir / "demo"
    bundle_dir = bundle_root / "variant_0"
    identity_map = json.loads((bundle_dir / "identity_map.json").read_text())
    symbol_map = json.loads((bundle_dir / "namespace_symbol_map.json").read_text())
    path_map = json.loads((bundle_dir / "namespace_path_map.json").read_text())
    meta = json.loads((bundle_dir / "mapping_meta.json").read_text())
    index = json.loads((bundle_root / "index.json").read_text())

    assert (bundle_dir / "identity_map.json").exists()
    assert (bundle_dir / "namespace_symbol_map.json").exists()
    assert (bundle_dir / "namespace_path_map.json").exists()
    assert index["variants"] == [{"name": "variant_0", "seed": 42, "variant_index": 0}]
    assert identity_map == {}
    assert "demo.query" in symbol_map
    assert "demo.subpkg" in symbol_map
    assert "demo.subpkg.fetcher" in symbol_map
    assert "query.py" in path_map
    assert meta["level"] == "namespace_l2"
    assert meta["mapping_version"] == "l2-v1"
    assert meta["repo_name"] == "demo"
    assert meta["seed"] == 42
    assert meta["mapping_scope"] == "repo"
    assert meta["namespace_granularity"] == "A"
    assert meta["enabled_layers"] == ["namespace_l2"]


def test_level2_bundle_generation_uses_level2_target_tokens_from_cache(tmp_path):
    repo_root = tmp_path / "repos"
    repo_dir = repo_root / "swe-bench_demo__demo-1"
    pkg = repo_dir / "demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from demo.query import build_query\n")
    (pkg / "query.py").write_text(
        "def build_query():\n"
        "    return 1\n"
    )

    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    cache_repo_dir = cache_dir / "demo"
    cache_repo_dir.mkdir(parents=True)
    (cache_repo_dir / "token_candidates.json").write_text(
        json.dumps({"build": ["forge"], "query": ["lookup"]})
    )

    script_path = Path(__file__).resolve().parents[2] / "glasses" / "main.py"
    project_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--instance-id",
            "demo__demo-1",
            "--seed",
            "42",
            "--mapping-mode",
            "namespace_l2",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_base_env(project_root),
        cwd=project_root,
    )

    bundle_dir = output_dir / "demo" / "variant_0"
    symbol_map = json.loads((bundle_dir / "namespace_symbol_map.json").read_text())

    assert symbol_map["build_query"] == "forge_lookup"


def test_identity_only_flag_keeps_legacy_cli_path(tmp_path):
    repo_root = tmp_path / "repos"
    repo_dir = repo_root / "swe-bench_demo__demo-1"
    pkg = repo_dir / "demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "query.py").write_text("def build_query():\n    return 1\n")

    output_dir = tmp_path / "output"
    script_path = Path(__file__).resolve().parents[2] / "glasses" / "main.py"
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_base_env(project_root),
        cwd=project_root,
    )

    assert "--identity-only" in result.stdout

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--instance-id",
            "demo__demo-1",
            "--seed",
            "42",
            "--identity-only",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_base_env(project_root),
        cwd=project_root,
    )

    bundle_root = output_dir / "demo"
    bundle_dir = bundle_root / "variant_0"
    identity_map = json.loads((bundle_dir / "identity_map.json").read_text())
    symbol_map = json.loads((bundle_dir / "namespace_symbol_map.json").read_text())
    path_map = json.loads((bundle_dir / "namespace_path_map.json").read_text())
    meta = json.loads((bundle_dir / "mapping_meta.json").read_text())
    index = json.loads((bundle_root / "index.json").read_text())

    assert identity_map == {"demo": "working_repository"}
    assert symbol_map == {}
    assert path_map == {}
    assert meta["level"] == "identity_only"
    assert meta["enabled_layers"] == ["identity_l1"]
    assert index["variants"] == [{"name": "variant_0", "seed": 42, "variant_index": 0}]


def test_level2_bundle_does_not_map_third_party_or_builtin_tokens(tmp_path):
    repo_root = tmp_path / "repos"
    repo_dir = repo_root / "swe-bench_demo__demo-1"
    pkg = repo_dir / "demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "adapter.py").write_text(
        "import requests\n"
        "from numpy import array\n\n"
        "def build_requests_adapter():\n"
        "    return requests.get('https://example.com')\n\n"
        "class ListManager:\n"
        "    pass\n"
    )

    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    cache_repo_dir = cache_dir / "demo"
    cache_repo_dir.mkdir(parents=True)
    (cache_repo_dir / "token_candidates.json").write_text(
        json.dumps(
            {
                "build": ["forge"],
                "requests": ["clientlib"],
                "adapter": ["bridge"],
                "list": ["arrayish"],
                "manager": ["overseer"],
            }
        )
    )

    script_path = Path(__file__).resolve().parents[2] / "glasses" / "main.py"
    project_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--instance-id",
            "demo__demo-1",
            "--seed",
            "42",
            "--mapping-mode",
            "namespace_l2",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_base_env(project_root),
        cwd=project_root,
    )

    symbol_map = json.loads((output_dir / "demo" / "variant_0" / "namespace_symbol_map.json").read_text())

    assert symbol_map["build_requests_adapter"] == "forge_requests_bridge"
    assert symbol_map["ListManager"] == "ListOverseer"


def test_level2_bundle_generation_ignores_shell_token_candidates(tmp_path):
    repo_root = tmp_path / "repos"
    repo_dir = repo_root / "swe-bench_demo__demo-1"
    pkg = repo_dir / "demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "query.py").write_text(
        "def sort_records():\n"
        "    return 1\n"
    )

    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    cache_repo_dir = cache_dir / "demo"
    cache_repo_dir.mkdir(parents=True)
    (cache_repo_dir / "token_candidates.json").write_text(json.dumps({"sort": ["arrange"]}))

    script_path = Path(__file__).resolve().parents[2] / "glasses" / "main.py"
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--instance-id",
            "demo__demo-1",
            "--seed",
            "42",
            "--mapping-mode",
            "namespace_l2",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        capture_output=True,
        text=True,
        env=_base_env(project_root),
        cwd=project_root,
    )

    assert result.returncode == 0

    token_mapping = json.loads((output_dir / "demo__demo-1" / "token_mapping.json").read_text())
    assert "sort" not in token_mapping


def test_identity_namespace_l2_keeps_repo_root_only_in_identity_map(tmp_path):
    repo_root = tmp_path / "repos"
    repo_dir = repo_root / "swe-bench_demo__demo-1"
    pkg = repo_dir / "demo"
    subpkg = pkg / "subpkg"
    subpkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "query.py").write_text(
        "class DemoThing:\n"
        "    pass\n\n"
        "def build_query():\n"
        "    return DemoThing()\n"
    )
    (subpkg / "__init__.py").write_text("")
    (subpkg / "worker.py").write_text("def nested_job():\n    return 1\n")

    output_dir = tmp_path / "output"
    script_path = Path(__file__).resolve().parents[2] / "glasses" / "main.py"
    project_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--instance-id",
            "demo__demo-1",
            "--seed",
            "42",
            "--mapping-mode",
            "identity_namespace_l2",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_base_env(project_root),
        cwd=project_root,
    )

    bundle_dir = output_dir / "demo" / "variant_0"
    identity_map = json.loads((bundle_dir / "identity_map.json").read_text())
    symbol_map = json.loads((bundle_dir / "namespace_symbol_map.json").read_text())
    path_map = json.loads((bundle_dir / "namespace_path_map.json").read_text())
    meta = json.loads((bundle_dir / "mapping_meta.json").read_text())

    assert identity_map == {"demo": "working_repository"}
    assert meta["level"] == "identity_namespace_l2"
    assert meta["enabled_layers"] == ["identity_l1", "namespace_l2"]
    assert "demo" not in symbol_map
    assert "demo" not in path_map


def test_level2_bundle_generation_writes_multiple_variants(tmp_path):
    repo_root = tmp_path / "repos"
    repo_dir = repo_root / "swe-bench_demo__demo-1"
    pkg = repo_dir / "demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "query.py").write_text(
        "class DemoThing:\n"
        "    pass\n\n"
        "def build_query():\n"
        "    return DemoThing()\n"
    )

    output_dir = tmp_path / "output"
    script_path = Path(__file__).resolve().parents[2] / "glasses" / "main.py"
    project_root = Path(__file__).resolve().parents[2]
    env = _base_env(project_root)

    for variant_index, seed in enumerate((42, 43)):
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--instance-id",
                "demo__demo-1",
                "--seed",
                str(seed),
                "--variant-index",
                str(variant_index),
                "--variant-count",
                "2",
                "--mapping-mode",
                "identity_namespace_l2",
                "--repo-root",
                str(repo_root),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=project_root,
        )

    bundle_root = output_dir / "demo"
    index = json.loads((bundle_root / "index.json").read_text())

    assert [item["name"] for item in index["variants"]] == ["variant_0", "variant_1"]
    assert [item["seed"] for item in index["variants"]] == [42, 43]
    assert (bundle_root / "variant_0" / "mapping_meta.json").exists()
    assert (bundle_root / "variant_1" / "mapping_meta.json").exists()
