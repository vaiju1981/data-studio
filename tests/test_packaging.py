"""What gets installed, and whether it is what was tested.

The lockfile is only worth having if the thing that ships uses it. CI installed
from it while the image resolved dependencies afresh from pyproject, so the
deployed set could differ from the tested one by any release published in
between — the one failure a lockfile exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def quoted_names(block: str) -> set[str]:
    """The package names in a quoted list, without a TOML parser: this env is 3.10
    and tomllib arrived in 3.11, which is not worth a dependency to read two lists."""
    return {re.split(r"[<>=!;\[ ]", item)[0].lower() for item in re.findall(r'"([^"]+)"', block)}


def declared_dependencies() -> set[str]:
    block = re.search(r"dependencies = \[(.*?)\]", (ROOT / "pyproject.toml").read_text(), re.S)
    return quoted_names(block.group(1))


def locked_packages() -> set[str]:
    lines = (ROOT / "requirements.lock").read_text().splitlines()
    return {line.split("==")[0].lower() for line in lines if "==" in line}


def test_the_image_installs_the_versions_that_were_tested() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "requirements.lock" in dockerfile, "the image resolves dependencies afresh"
    assert "--no-deps" in dockerfile, (
        "installing the wheel with its dependencies lets pip resolve past the lockfile"
    )


def test_every_declared_dependency_is_pinned() -> None:
    missing = declared_dependencies() - locked_packages()
    assert not missing, f"declared in pyproject but absent from requirements.lock: {missing}"


def test_the_runtime_roots_and_the_dependencies_are_the_same_list() -> None:
    """Two lists that must agree, in two files. A root missing from pyproject is
    pinned without being required; a dependency missing from the roots is required
    without being pinned, and only the second is visible at install time."""
    roots = re.search(r"RUNTIME = \[(.*?)\]", (ROOT / "tools/lock.py").read_text(), re.S)
    # httpx is Ollama's transport and arrives through its closure; it is declared
    # because the retry catches its timeout by name, not because it is a root.
    assert quoted_names(roots.group(1)) == declared_dependencies() - {"httpx"}


def test_nothing_is_declared_that_the_application_never_imports() -> None:
    """scikit-learn sat in pyproject unimported, pinning joblib and threadpoolctl
    behind it into every image. A dependency earns its place by being used."""
    package = ROOT / "src/smart_data_studio"
    imports = "\n".join(
        line
        for path in package.rglob("*.py")
        for line in path.read_text().splitlines()
        if line.startswith(("import ", "from "))
    )
    # Distribution name to import name, for the ones that differ. Empty today, and
    # kept so re-adding such a package is a deliberate line rather than a puzzle.
    IMPORTED_AS: dict[str, str] = {}
    unused = [
        name
        for name in declared_dependencies()
        if not re.search(
            rf"^(import|from) {re.escape(IMPORTED_AS.get(name, name))}\b", imports, re.M
        )
    ]
    assert not unused, f"declared in pyproject but never imported: {sorted(unused)}"
