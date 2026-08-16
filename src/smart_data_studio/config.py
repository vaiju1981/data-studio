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
SAMPLE_ROWS = 5

MAX_TOOL_ROUNDS = 6
MAX_EXPLORE_ROUNDS = 8

# Older tool results are replaced by a placeholder once this many newer ones exist.
KEEP_TOOL_PAYLOADS = 4
