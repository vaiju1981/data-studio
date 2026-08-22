"""Regenerate requirements.lock from the current environment.

Pins the closure of what the application actually imports, not the whole
development environment, so the lockfile describes the runtime and nothing else.

    python tools/lock.py
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
import subprocess
import sys
from pathlib import Path

RUNTIME = [
    "duckdb",
    "ollama",
    "pandas",
    "plotly",
    "scipy",
    "sqlglot",
    "statsmodels",
    "streamlit",
]


def installed_versions() -> dict[str, str]:
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    return dict(
        line.split("==", 1) for line in frozen.splitlines() if "==" in line and " @ " not in line
    )


def closure(roots: list[str]) -> set[str]:
    """Every package the roots pull in, ignoring optional extras."""
    seen: set[str] = set()
    queue = list(roots)
    while queue:
        name = queue.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        try:
            for requirement in metadata.requires(name) or []:
                if "extra ==" in requirement:
                    continue
                dependency = re.split(r"[<>=!;\[ ]", requirement.strip())[0]
                if dependency:
                    queue.append(dependency)
        except metadata.PackageNotFoundError:
            pass
    return seen


def main() -> int:
    needed = closure(RUNTIME)
    pinned = sorted(
        f"{name}=={version}"
        for name, version in installed_versions().items()
        if name.lower().replace("_", "-") in needed
    )
    Path("requirements.lock").write_text(
        "# Resolved runtime closure, regenerate with tools/lock.py after changing pyproject.\n"
        "# Pinned so a build is reproducible; CI installs from here.\n" + "\n".join(pinned) + "\n"
    )
    print(f"pinned {len(pinned)} packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
