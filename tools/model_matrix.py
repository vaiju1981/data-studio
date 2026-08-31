"""Run one bank against several models and print the rates side by side.

    python tools/model_matrix.py tests/test_domain_bank.py glm-5.2:cloud kimi-k3:cloud

Nothing in the code depends on which model is served — two config values decide
it — but the *answer rate* depends on it entirely, and that had never been
measured. A deployer choosing a model has been choosing blind.

The domain bank is the one to run here: its five workspaces are built from a
seed, so a result can be reproduced by anyone and published without exposing
anybody's data. The casino banks measure the same thing against files that cannot
leave the machine.

Each model is probed for tool calling before its run, because the agent is a tool
loop: a model that cannot call one produces thirteen identical failures and an
hour of wondering why.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROBE = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]


def calls_tools(model: str) -> str | None:
    """None when the model called the tool, else why it cannot be measured."""
    import ollama

    host = os.environ.get("SDS_OLLAMA_HOST", "http://localhost:11434")
    try:
        reply = ollama.Client(host=host).chat(
            model=model,
            messages=[{"role": "user", "content": "What is 21 plus 21? Use the tool."}],
            tools=PROBE,
            options={"num_predict": 128},
        )
    except Exception as error:
        return f"{type(error).__name__}: {error}"[:120]
    if not (reply.get("message", {}).get("tool_calls") or []):
        return "served, but did not call the tool it was offered"
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip())
        return 2
    bank, models = argv[1], argv[2:]
    results = ROOT / "matrix-results.jsonl"
    results.unlink(missing_ok=True)

    for model in models:
        refusal = calls_tools(model)
        if refusal:
            # Named and skipped rather than run: the agent is a tool loop, and a
            # row of zeroes would read as a bad model rather than an absent one.
            print(f"skipping {model}: {refusal}")
            continue
        print(f"running {bank} against {model}…", flush=True)
        environment = {
            **os.environ,
            "USE_LLM": "1",
            "SDS_MODEL_ID": model,
            "SDS_BANK_RESULTS": str(results),
        }
        finished = subprocess.run(
            [sys.executable, "-m", "pytest", bank, "-q"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        print(f"  {finished.stdout.strip().splitlines()[-1] if finished.stdout else 'no output'}")

    if not results.exists():
        print("\nNo model produced a result.")
        return 1
    print()
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/bank_report.py"), str(results)], cwd=ROOT
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
