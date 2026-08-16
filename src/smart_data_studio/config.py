"""Runtime settings kept small enough to audit at a glance.

Anything a deployment must change is read from the environment, so a hosted
instance is configured rather than edited.
"""

import os
from pathlib import Path

VERSION = "0.1.0"
# Bumped whenever the analyst prompt changes, so a logged answer can be traced to
# the instructions that produced it.
PROMPT_VERSION = "2026-08-16.1"


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _number(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


MODEL_ID = os.environ.get("SDS_MODEL_ID", "gemma4:31b-cloud")
OLLAMA_HOST = os.environ.get("SDS_OLLAMA_HOST", "http://localhost:11434")

# Reading a path off the host is right for a local tool and disqualifying for a
# shared one, so hosted deployments turn it off.
ALLOW_LOCAL_PATHS = _flag("SDS_ALLOW_LOCAL_PATHS", True)

# DuckDB is given a budget before the connection is locked; without one a single
# careless join can take the whole process down with it.
DUCKDB_MEMORY_LIMIT = os.environ.get("SDS_DUCKDB_MEMORY_LIMIT", "4GB")
DUCKDB_THREADS = _number("SDS_DUCKDB_THREADS", 4)
DUCKDB_TEMP_DIR = os.environ.get("SDS_DUCKDB_TEMP_DIR", "")
QUERY_TIMEOUT_SECONDS = _number("SDS_QUERY_TIMEOUT_SECONDS", 60)

# Upload ceilings, checked before parsing rather than after.
MAX_UPLOAD_BYTES = _number("SDS_MAX_UPLOAD_BYTES", 500 * 1024 * 1024)
MAX_INGEST_ROWS = _number("SDS_MAX_INGEST_ROWS", 20_000_000)
MAX_INGEST_COLUMNS = _number("SDS_MAX_INGEST_COLUMNS", 512)

LOG_LEVEL = os.environ.get("SDS_LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get("SDS_LOG_FORMAT", "json")


def temp_directory() -> str:
    return DUCKDB_TEMP_DIR or str(Path(os.environ.get("TMPDIR", "/tmp")) / "smart-data-studio")


# Three separate ceilings. They were one value before, which silently truncated
# downloads to whatever the model happened to be allowed to read.
MAX_LLM_ROWS = 200  # most rows the model may ever see at once
MAX_DISPLAY_ROWS = 5_000  # rows kept for the table and the chart
MAX_EXPORT_ROWS = 250_000  # ceiling for an on-demand CSV export

# Above this many characters the model gets a digest instead of the rows. A wide
# 200-row result costs ~64k tokens; a typical aggregate costs under 100.
MAX_LLM_PAYLOAD_CHARS = 12_000
DIGEST_SAMPLE_ROWS = 10

MAX_CHART_ROWS = 5_000
# Statistical tools read the whole result, so this bounds what they will pull.
# Budgeted in cells rather than rows: a two-column comparison can hold every row
# of a large table, while a 57-column result cannot, and one row cap cannot serve
# both without either sampling needlessly or running out of memory.
MAX_ANALYSIS_CELLS = 20_000_000
MAX_ANALYSIS_ROWS = 2_000_000
# Fixed so the same question gives the same answer twice.
ANALYSIS_SAMPLE_SEED = 42
SAMPLE_ROWS = 5

# Statistical tests get their power from sample size; past this the extra rows only
# cost time, and effect size — the part that matters — is already stable.
MAX_TEST_SAMPLE = 50_000
MAX_RELATE_SAMPLE = 100_000
# A column with more levels than this is an identifier, not a dimension to sweep.
MAX_DRIVER_LEVELS = 50
# Below this a group cannot support a test: Cliff's delta from one observation is
# 1.0, which reads as a maximal effect.
MIN_COMPARISON_ROWS = 10
MIN_ASSOCIATION_ROWS = 10
# A forecast cannot reach further ahead than the history it was built from.
MAX_FORECAST_PERIODS = 120

# Eight tools means longer chains; six rounds left complex questions unanswered.
MAX_TOOL_ROUNDS = 10
MAX_EXPLORE_ROUNDS = 8

# Older tool results are replaced by a placeholder once this many newer ones exist.
KEEP_TOOL_PAYLOADS = 4
