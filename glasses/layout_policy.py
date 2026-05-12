from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RepoLayoutPolicy:
    repo_name: str
    protected_prefixes: tuple[str, ...]
    preferred_source_prefixes: tuple[str, ...] = ()
    excluded_target_prefixes: tuple[str, ...] = ()
    protected_filenames: tuple[str, ...] = ()
    protected_path_fragments: tuple[str, ...] = ()
    namespace_root: str | None = None
    min_primary_score: float = 8.0
    min_support_score: float = 3.0
    max_source_files: int = 4
    max_target_dirs: int = 5


DEFAULT_POLICY = RepoLayoutPolicy(
    repo_name="default",
    protected_prefixes=(
        ".github/",
        "benchmarks/",
        "build/",
        "ci/",
        "docs/",
        "doc/",
        "examples/",
        "locale/",
        "migrations/",
        "scripts/",
        "static/",
        "templates/",
        "test/",
        "tests/",
        "testing/",
    ),
    protected_filenames=(
        "__init__.py",
        "conftest.py",
        "setup.py",
        "noxfile.py",
        "tasks.py",
        "versioneer.py",
    ),
)


REPO_POLICIES = {
    "default": DEFAULT_POLICY,
    "django": RepoLayoutPolicy(
        repo_name="django",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "django/conf/locale/",
            "django/contrib/admin/static/",
            "django/contrib/admin/templates/",
        ),
        excluded_target_prefixes=(
            "django/core/management/commands/",
            "django/db/migrations/",
        ),
        namespace_root="django",
        preferred_source_prefixes=(
            "django/core/",
            "django/db/",
            "django/forms/",
            "django/template/",
            "django/utils/",
            "django/contrib/",
        ),
    ),
    "matplotlib": RepoLayoutPolicy(
        repo_name="matplotlib",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "lib/mpl_toolkits/",
        ),
        protected_path_fragments=(
            "/backends/",
        ),
        namespace_root="lib/matplotlib",
        preferred_source_prefixes=("lib/matplotlib/",),
    ),
    "astropy": RepoLayoutPolicy(
        repo_name="astropy",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "cextern/",
        ),
        protected_path_fragments=(
            "/config/",
            "/wcs/",
            "/io/fits/",
        ),
        namespace_root="astropy",
        preferred_source_prefixes=("astropy/",),
    ),
    "scikit-learn": RepoLayoutPolicy(
        repo_name="scikit-learn",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "benchmarks/",
            "examples/",
            "sklearn/externals/",
            "sklearn/__check_build/",
        ),
        protected_path_fragments=(
            "/externals/",
        ),
        namespace_root="sklearn",
        preferred_source_prefixes=("sklearn/",),
    ),
    "pydata": RepoLayoutPolicy(
        repo_name="pydata",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "asv_bench/",
            "properties/",
        ),
        namespace_root="xarray",
        preferred_source_prefixes=("xarray/",),
    ),
    "pytest-dev": RepoLayoutPolicy(
        repo_name="pytest-dev",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "bench/",
            "extra/",
        ),
        namespace_root="src",
        preferred_source_prefixes=("src/_pytest/", "src/pytest/"),
    ),
    "pylint-dev": RepoLayoutPolicy(
        repo_name="pylint-dev",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "elisp/",
            "script/",
        ),
        namespace_root="pylint",
        preferred_source_prefixes=("pylint/",),
    ),
    "psf": RepoLayoutPolicy(
        repo_name="psf",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "ext/",
        ),
        namespace_root="requests",
        preferred_source_prefixes=("requests/",),
    ),
    "pallets": RepoLayoutPolicy(
        repo_name="pallets",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "requirements/",
            "artwork/",
        ),
        namespace_root="src/flask",
        preferred_source_prefixes=("src/flask/",),
    ),
    "mwaskom": RepoLayoutPolicy(
        repo_name="mwaskom",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes,
        namespace_root="seaborn",
        preferred_source_prefixes=("seaborn/",),
    ),
    "sphinx-doc": RepoLayoutPolicy(
        repo_name="sphinx-doc",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes
        + (
            "utils/",
        ),
        namespace_root="sphinx",
        preferred_source_prefixes=("sphinx/",),
    ),
    "sympy": RepoLayoutPolicy(
        repo_name="sympy",
        protected_prefixes=DEFAULT_POLICY.protected_prefixes,
        namespace_root="sympy",
        preferred_source_prefixes=("sympy/",),
    ),
}


def infer_repo_name(instance_id: str) -> str:
    return instance_id.split("__", 1)[0]


def get_repo_policy(repo_name: str) -> RepoLayoutPolicy:
    policy = REPO_POLICIES.get(repo_name, DEFAULT_POLICY)
    if policy is DEFAULT_POLICY:
        return policy
    return replace(
        policy,
        protected_filenames=policy.protected_filenames or DEFAULT_POLICY.protected_filenames,
        protected_path_fragments=policy.protected_path_fragments or DEFAULT_POLICY.protected_path_fragments,
    )


def is_protected_path(path: str, policy: RepoLayoutPolicy) -> bool:
    normalized = path.lstrip("./")
    filename = normalized.rsplit("/", 1)[-1]
    if filename in policy.protected_filenames:
        return True
    for fragment in policy.protected_path_fragments:
        if fragment in f"/{normalized}":
            return True
    for prefix in policy.protected_prefixes + policy.excluded_target_prefixes:
        if normalized.startswith(prefix):
            return True
        if prefix.startswith("*/") and f"/{prefix[2:]}" in f"/{normalized}":
            return True
    return False


def prefers_source_path(path: str, policy: RepoLayoutPolicy) -> bool:
    normalized = path.lstrip("./")
    if not policy.preferred_source_prefixes:
        return True
    return any(normalized.startswith(prefix) for prefix in policy.preferred_source_prefixes)
