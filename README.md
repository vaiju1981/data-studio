# Smart Data Studio

Load one or more data files — CSV, TSV, Parquet, JSON, NDJSON or Excel, plain or
zipped — get an automatic profile and written insights, then ask questions in
plain English and get back tables, answers and charts.

Every data answer shows the SQL and the result row count that produced it. The
model writes SQL against an in-memory DuckDB workspace that is locked read-only
before it sees a single query.

## Run

The project uses the `data-studio` conda environment and an Ollama endpoint of
your choosing — see [Choosing a model](#choosing-a-model).

```bash
conda activate data-studio
python -m pip install -e .
streamlit run src/smart_data_studio/ui/app.py
```

Open the local URL Streamlit prints. Upload files in the sidebar, or enter one
local path per line — paths are resolved on the machine running Streamlit. Load
several related files and the model can join across them, whatever mix of formats
they arrive in. Each sheet of an Excel workbook becomes its own table.

## How it works

- **Load** — files become DuckDB tables — Parquet, JSON and compressed CSV
  through DuckDB's own readers, Excel a sheet at a time — then the connection is locked
  (`enable_external_access=false`, `lock_configuration=true`) so model-written SQL
  cannot reach the filesystem. `sqlglot` separately rejects anything that is not a
  single `SELECT` over the loaded tables.
- **Understand** — before the first question the model explores the data with real
  queries and folds what it learns into the chat context, alongside the column
  profile and sample rows.
- **Ask** — a tool loop over twelve tools: `run_sql` and `make_chart`, plus
  `forecast`, `analyze_trend`, `detect_anomalies`, `compare_groups`,
  `compare_rates`, `rank_drivers`, `relate`, `find_outliers` and `cohort_window`, which run real
  statistics rather than asking the model to estimate them. Charts are built from a validated spec, never from
  model-written plotting code. `find_values` resolves a name to the values a
  column actually holds before anything filters on it — `NORTH LAS VEGAS` and
  `N LAS VEGAS` are one city, and matching only the first undercounts by a fifth
  while returning a number that looks right.
- **Investigate** — a question of judgement is planned into sub-questions first,
  each worked separately before a final pass answers from all of them. Two rules
  do the work: ask the question at the grain the decision is made at, and include
  one sub-question that could show the obvious answer to be wrong. Asked how to
  upsell a tier, the single-pass answer averaged per visit and called the local
  segment weak; per player it is worth three times either alternative. A lookup
  is answered in one pass either way.

Local paths you have loaded before are offered as a list, so a file is picked
rather than retyped, and a remembered file and a newly typed one load together.
This is the only thing written to disk — paths, never data, in
`~/.smart-data-studio/recent.json`, cleared by **Delete my data**.

Two conversation modes: **multi-turn** lets the model see earlier questions and
results, so follow-ups like "now chart that" work; **single turn** answers each
question from the data alone. The full history stays on screen either way, and
the sidebar can force investigation on or off.

Large results never reach the prompt verbatim. Past a size budget the model
receives a digest — column types, exact row count, and statistics computed over
the whole result — while the table and the export keep every row.

## Running it as a service

```bash
docker build -t smart-data-studio .
docker run -p 8501:8501 --read-only --tmpfs /tmp -v sds-work:/workspace \
  -e SDS_OLLAMA_HOST=http://ollama:11434 smart-data-studio
```

The image runs as a non-root user, keeps all writable state under `/workspace`,
and turns local-path loading **off** — a hosted instance should not read arbitrary
files from its host. Liveness is Streamlit's own `/_stcore/health`; readiness is
`python -m smart_data_studio.healthcheck`, which additionally proves DuckDB works
and that the configured model is actually served by the configured endpoint.

### Configuration

Everything a deployment needs to change is an environment variable, so the image
is configured rather than edited.

| Variable | Default | Purpose |
|---|---|---|
| `SDS_MODEL_ID`, `SDS_OLLAMA_HOST` | `gemma4:31b-cloud`, `localhost:11434` | Which model, served from where |
| `SDS_ALLOW_LOCAL_PATHS` | `true` (`false` in the image) | Read data files from the host filesystem |
| `SDS_SENSITIVE_COLUMNS` | *(empty)* | Comma-separated names; matching columns are withheld from everything the model sees |
| `SDS_DUCKDB_MEMORY_LIMIT`, `SDS_DUCKDB_THREADS` | `4GB`, `4` | Query budget, applied before the connection locks |
| `SDS_DUCKDB_TEMP_LIMIT` | `20GB` | What one workspace may spill to disk. DuckDB's own default is 90% of the volume |
| `SDS_TIME_ZONE` | `UTC` | The zone timestamps carrying an offset are read in. Unpinned, DuckDB takes it from the host and the same question answers differently on two machines |
| `SDS_QUERY_TIMEOUT_SECONDS` | `60` | A query past this is interrupted; the session survives |
| `SDS_MODEL_TIMEOUT_SECONDS` | `300` | A model call past this is abandoned and retried; without one a stalled completion holds its workspace forever |
| `SDS_MAX_UPLOAD_BYTES`, `SDS_MAX_INGEST_ROWS`, `SDS_MAX_INGEST_COLUMNS` | 500MB, 20M, 512 | Upload ceilings, checked before parsing |
| `SDS_MAX_UPLOAD_TOTAL_BYTES`, `SDS_MAX_FILES_PER_LOAD` | 1GB, `20` | Ceilings on a whole load, checked before a byte is read |
| `SDS_MAX_ACTIVE_SESSIONS`, `SDS_SESSION_IDLE_SECONDS` | `8`, `3600` | Concurrent workspaces, and when an idle one is reclaimed. A workspace answering a question is never idle |
| `SDS_LOG_FORMAT`, `SDS_LOG_LEVEL` | `json`, `INFO` | Structured logs on stdout |

**Sizing.** Each session holds its own DuckDB, bounded by
`SDS_DUCKDB_MEMORY_LIMIT` rather than by the size of the file: past that budget
DuckDB offloads table data to `SDS_DUCKDB_TEMP_DIR`, so a file larger than the
budget loads and queries rather than failing. Size the host by the two budgets
multiplied by `SDS_MAX_ACTIVE_SESSIONS` — RAM for the first, disk for the second
— and the ninth user is turned away rather than the first eight being starved.

### Operating

- **Logs** are one JSON object per event on stdout, carrying session and question
  ids, the app and prompt versions, and timings for ingest, queries, tool calls and
  model calls. No cell value is logged: the SQL is, because it is the evidence
  behind an answer, but every literal, comment and select alias in it is masked
  first — the three ways a value rides in, since `AS North` is as much a cell as
  `WHERE region = 'North'`. The log is the one stream that leaves the host; the
  query itself is on screen beside the answer with everything intact.
- **Rollback** is redeploying the previous image tag; the app holds no durable
  state, so nothing migrates and nothing needs restoring. That is also the backup
  story: uploads and results live only in the running process and the mounted
  `/workspace`, and both are discarded on stop.
- **Model outage** degrades to the profile: loading and the data profile still
  work, and the readiness check reports which half is down.

## Checks

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

The live-model regression replays the whole question bank against a real dataset.
It is slow and needs a model endpoint, so it is opt-in:

```bash
USE_LLM=1 pytest tests/test_question_bank.py -q
```

Regenerate the pinned runtime after changing dependencies:

```bash
python tools/lock.py
```

## Choosing a model

Two settings in `src/smart_data_studio/config.py` decide what you run against:

```python
MODEL_ID = os.environ.get("SDS_MODEL_ID", "gemma4:31b-cloud")
OLLAMA_HOST = os.environ.get("SDS_OLLAMA_HOST", "http://localhost:11434")
```

Set `SDS_OLLAMA_HOST` to wherever your Ollama runs — this machine, a server you
host, or a hosted Ollama endpoint — and `SDS_MODEL_ID` to any model that endpoint
serves, or edit the defaults above. The prefix matters: the ollama client reads
`OLLAMA_HOST` itself, so the unprefixed name would leave two things reading one
variable. Those two lines are the only place a model or provider is named;
nothing else in the code depends on either.

**Tool calling is the one requirement**, because the agent is a tool loop and
cannot run a query without it. Ollama lists tool-capable models at
<https://ollama.com/search?c=tools>; confirm a particular one with
`ollama show <model>` and look for `tools` under `Capabilities`.

Where your data goes follows from that endpoint: a model you run yourself keeps
schema, profile statistics and query results on your own machine, while a hosted
endpoint receives them.
