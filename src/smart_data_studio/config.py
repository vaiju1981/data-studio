"""Runtime settings kept small enough to audit at a glance."""

MODEL_ID = "gemma4:31b-cloud"
OLLAMA_HOST = "http://localhost:11434"

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

# Eight tools means longer chains; six rounds left complex questions unanswered.
MAX_TOOL_ROUNDS = 10
MAX_EXPLORE_ROUNDS = 8

# Older tool results are replaced by a placeholder once this many newer ones exist.
KEEP_TOOL_PAYLOADS = 4
