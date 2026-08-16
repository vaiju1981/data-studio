# Smart Data Studio

Load one or more CSV files, get an automatic profile and written insights, then
ask questions in plain English and get back tables, answers and charts.

Every data answer shows the SQL and the result row count that produced it. The
model writes SQL against an in-memory DuckDB workspace that is locked read-only
before it sees a single query.

## Run

The project uses the `data-studio` conda environment and a local Ollama server.
The configured model is `gemma4:31b-cloud`.

```bash
conda activate data-studio
python -m pip install -e .
streamlit run src/smart_data_studio/ui/app.py
```

Open the local URL Streamlit prints. Upload CSV files in the sidebar, or enter one
local path per line — paths are resolved on the machine running Streamlit. Load
several related files and the model can join across them.

## How it works

- **Load** — CSVs become DuckDB tables, then the connection is locked
  (`enable_external_access=false`, `lock_configuration=true`) so model-written SQL
  cannot reach the filesystem. `sqlglot` separately rejects anything that is not a
  single `SELECT` over the loaded tables.
- **Understand** — before the first question the model explores the data with real
  queries and folds what it learns into the chat context, alongside the column
  profile and sample rows.
- **Ask** — a tool loop with two tools, `run_sql` and `make_chart`. Charts are
  built from a validated spec, never from model-written plotting code.

Two conversation modes: **multi-turn** lets the model see earlier questions and
results, so follow-ups like "now chart that" work; **single turn** answers each
question from the data alone. The full history stays on screen either way.

Large results never reach the prompt verbatim. Past a size budget the model
receives a digest — column types, exact row count, and statistics computed over
the whole result — while the table and the export keep every row.

## Checks

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

## Note on the model

`gemma4:31b-cloud` runs on Ollama's infrastructure rather than locally. Schema
information, profile statistics and query results used in chat leave the machine.
The model id is a single value in `config.py`, so pointing at a fully local model
is a one-line change.
