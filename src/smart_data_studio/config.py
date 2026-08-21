"""Runtime settings kept small enough to audit at a glance.

Anything a deployment must change is read from the environment, so a hosted
instance is configured rather than edited.
"""

import os
from pathlib import Path

VERSION = "0.1.0"
# Bumped whenever the analyst prompt changes, so a logged answer can be traced to
# the instructions that produced it.
PROMPT_VERSION = "2026-08-18.3"


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _number(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


MODEL_ID = os.environ.get("SDS_MODEL_ID", "gemma4:31b-cloud")
OLLAMA_HOST = os.environ.get("SDS_OLLAMA_HOST", "http://localhost:11434")
# A hosted model returns the occasional 500. Cheap to retry, and an investigation
# makes six to ten calls, so a blip would otherwise discard a minute of work.
MODEL_RETRIES = _number("SDS_MODEL_RETRIES", 2)
MODEL_RETRY_SECONDS = _number("SDS_MODEL_RETRY_SECONDS", 2)
# A prompt past the context window is refused rather than truncated, so the
# conversation is shed and sent again. Three rounds clears the largest history the
# session limits allow.
MAX_CONTEXT_SHEDS = _number("SDS_MAX_CONTEXT_SHEDS", 3)

# Right for a local tool, disqualifying for a shared one, so hosted deployments
# turn it off.
ALLOW_LOCAL_PATHS = _flag("SDS_ALLOW_LOCAL_PATHS", True)

# DuckDB is given a budget before the connection is locked; without one a single
# careless join can take the whole process down with it.
DUCKDB_MEMORY_LIMIT = os.environ.get("SDS_DUCKDB_MEMORY_LIMIT", "4GB")
DUCKDB_THREADS = _number("SDS_DUCKDB_THREADS", 4)
DUCKDB_TEMP_DIR = os.environ.get("SDS_DUCKDB_TEMP_DIR", "")
QUERY_TIMEOUT_SECONDS = _number("SDS_QUERY_TIMEOUT_SECONDS", 60)

# Size ceilings, checked before parsing. A local path gets its own and a looser
# one: nothing crossed a network, and multi-gigabyte local files load and query
# perfectly well, so the upload limit would refuse the case paths exist for.
MAX_UPLOAD_BYTES = _number("SDS_MAX_UPLOAD_BYTES", 500 * 1024 * 1024)
MAX_LOCAL_FILE_BYTES = _number("SDS_MAX_LOCAL_FILE_BYTES", 5 * 1024 * 1024 * 1024)

# Shape ceilings, necessarily measured once the table is built.
MAX_INGEST_ROWS = _number("SDS_MAX_INGEST_ROWS", 20_000_000)
MAX_INGEST_COLUMNS = _number("SDS_MAX_INGEST_COLUMNS", 512)

# Deliberately loose: the binding constraint is DUCKDB_MEMORY_LIMIT, which spills
# to disk rather than failing, so this only catches the absurd. Tightened to 400M
# it refused ordinary wide files that load and query fine.
MAX_INGEST_CELLS = _number("SDS_MAX_INGEST_CELLS", 2_000_000_000)
MAX_HEADER_LENGTH = _number("SDS_MAX_HEADER_LENGTH", 200)
# Long free text is rarely the analysis and always the cost, so it is trimmed on
# the way to the model rather than on the way into the table.
MAX_CELL_CHARS_TO_MODEL = _number("SDS_MAX_CELL_CHARS_TO_MODEL", 200)

# Column names matching any of these are kept out of everything the model sees.
# Comma separated, matched case-insensitively as substrings.
SENSITIVE_COLUMNS = tuple(
    part.strip().lower()
    for part in os.environ.get("SDS_SENSITIVE_COLUMNS", "").split(",")
    if part.strip()
)

# Per-session ceilings, so one conversation cannot consume the host.
MAX_SESSION_QUERIES = _number("SDS_MAX_SESSION_QUERIES", 500)
MAX_SESSION_EXPORT_BYTES = _number("SDS_MAX_SESSION_EXPORT_BYTES", 500 * 1024 * 1024)
MAX_ACTIVE_SESSIONS = _number("SDS_MAX_ACTIVE_SESSIONS", 8)
SESSION_IDLE_SECONDS = _number("SDS_SESSION_IDLE_SECONDS", 3600)

# Generous enough that real analytics never trips them, tight enough that a
# runaway generated join is refused before it starts rather than after a minute.
MAX_QUERY_TABLES = _number("SDS_MAX_QUERY_TABLES", 12)
MAX_QUERY_DEPTH = _number("SDS_MAX_QUERY_DEPTH", 12)

LOG_LEVEL = os.environ.get("SDS_LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get("SDS_LOG_FORMAT", "json")


def temp_directory() -> str:
    """Where DuckDB spills, always inside a directory this app owns.

    The name is appended even to a configured path. Shutdown removes this
    directory recursively, and returning the configured path unchanged meant
    SDS_DUCKDB_TEMP_DIR=/data deleted /data — everything in it, not only the spill
    files we put there.
    """
    root = DUCKDB_TEMP_DIR or os.environ.get("TMPDIR", "/tmp")
    return str(Path(root) / "smart-data-studio")


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
# Budgeted in cells rather than rows: a two-column comparison can hold every row of
# a large table while a very wide result cannot, and one row cap cannot serve both.
MAX_ANALYSIS_CELLS = 20_000_000
MAX_ANALYSIS_ROWS = 2_000_000
# Fixed so the same question gives the same answer twice.
ANALYSIS_SAMPLE_SEED = 42
SAMPLE_ROWS = 5

# Statistical tests get their power from sample size; past this the extra rows only
# cost time, and effect size — the part that matters — is already stable.
MAX_TEST_SAMPLE = 50_000
MAX_RELATE_SAMPLE = 100_000
# Group summaries a comparison returns, listed largest first so the two actually
# compared always fall inside it. Uncapped, a high-cardinality dimension overruns
# MAX_LLM_PAYLOAD_CHARS many times over.
MAX_COMPARISON_GROUPS = 25
# Values listed per dimension column. A column with no more than this many is
# listed in full; a wider one shows its commonest, which is what makes a column
# the model never queried still visible to it.
DICTIONARY_VALUES = 12

# Values returned when looking up what a column holds. Enough to show every
# spelling of a name; short enough not to dump a column into the prompt.
MAX_VALUE_MATCHES = 25

# Bounds on what the model may propose about how tables relate: every candidate
# costs a verification query.
MAX_KEY_CANDIDATES = 4
MAX_JOIN_CANDIDATES = 4
MAX_KEY_COLUMNS = 4
# How many of a table's most various columns are searched for a key. Pairs are
# quadratic, and a key's parts are necessarily among the most various columns, so
# a small window finds a composite key without a full sweep.
MAX_KEY_SEARCH_COLUMNS = 6
# Shared column names reported per table: two related files can share dozens, and
# listing them all buries the few a join would use.
MAX_SHARED_COLUMNS = 5

# Columns whose value changes within an entity, named in the profile. Taken from
# both ends of the range rather than the top: a column changing for many entities
# is a per-row value, one changing for a handful looks like a property of the
# entity and is not, and only the second is something a reader cannot infer. The
# middle is the least informative part, so that is what the cap drops.
MAX_VARYING_COLUMNS = 8

# A column with more levels than this is an identifier, not a dimension to sweep.
MAX_DRIVER_LEVELS = 50
# Below this a group cannot support a test: Cliff's delta from one observation is
# 1.0, which reads as a maximal effect.
MIN_COMPARISON_ROWS = 10
MIN_ASSOCIATION_ROWS = 10
# A repeated extreme is a missing-value code, not a measurement. Set low because
# the consequence — an average computed over sentinels — is silent and severe.
SENTINEL_SHARE = 0.02
MIN_SENTINEL_ROWS = 50
# How many typical steps the value must stand away from the rest before it is a
# code rather than a reading. Twenty leaves a binned column alone.
SENTINEL_GAP_RATIO = 20

# A forecast cannot reach further ahead than the history it was built from.
MAX_FORECAST_PERIODS = 120

# Eight tools means longer chains than six rounds can finish.
MAX_TOOL_ROUNDS = 10
MAX_EXPLORE_ROUNDS = 8
# A judgement question is worked as a few sub-questions, each on a short leash.
MAX_PLAN_STEPS = 5
MAX_STEP_ROUNDS = 5

# Older tool results are replaced by a placeholder once this many newer ones exist.
KEEP_TOOL_PAYLOADS = 4
