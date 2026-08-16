"""Readiness check: can this instance actually answer, not merely is it running.

Used as the container HEALTHCHECK and runnable by hand:

    python -m smart_data_studio.healthcheck

It never sends user data anywhere. The model check asks Ollama which models it
serves; it does not run a completion.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

import duckdb

from smart_data_studio.config import MODEL_ID, OLLAMA_HOST, VERSION


def check_duckdb() -> tuple[bool, str]:
    try:
        with duckdb.connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return True, f"duckdb {duckdb.__version__}"
    except Exception as error:  # pragma: no cover - only on a broken install
        return False, f"duckdb unavailable: {error}"


def check_model(timeout: float = 3.0) -> tuple[bool, str]:
    """Is the configured model actually served by the configured endpoint?

    A reachable Ollama with the wrong model is the failure that otherwise shows up
    as a puzzling error on the user's first question.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=timeout) as response:
            served = {model["name"] for model in json.load(response).get("models", [])}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
        return False, f"{OLLAMA_HOST} unreachable: {error}"
    if MODEL_ID not in served:
        return False, f"{MODEL_ID} is not served by {OLLAMA_HOST}"
    return True, f"{MODEL_ID} available"


def main() -> int:
    checks = {"duckdb": check_duckdb(), "model": check_model()}
    ready = all(ok for ok, _ in checks.values())
    print(
        json.dumps(
            {
                "ready": ready,
                "version": VERSION,
                **{name: detail for name, (_, detail) in checks.items()},
            }
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
