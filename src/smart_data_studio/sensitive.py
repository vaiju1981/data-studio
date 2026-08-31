"""Which columns hold personal data, proposed before the data is loaded.

`SDS_SENSITIVE_COLUMNS` is an operator's setting: it is decided once, in a
deployment, by somebody who has not seen the file. The person uploading has seen
it and knows, and until now had no way to say so — which made the setting a
mechanism only its author ever used, and left `birthDate` and `zipCode` going to
a hosted model because nobody had thought to name them in advance.

So the model is asked, from **column names and types only**. That ordering is the
whole point: this runs before any table is built, so nothing has read a value,
and the proposal cannot itself be the leak it exists to prevent.

It proposes. The person confirms. Two rules follow from that:

- the deterministic name rule is a floor, not a suggestion — a model that misses
  `birthDate` cannot make it un-personal;
- what the operator configured is not up for discussion, because a deployment
  policy that a user can untick is not a policy.
"""

from __future__ import annotations

import json

import ollama

from smart_data_studio import logs
from smart_data_studio.config import (
    MODEL_ID,
    MODEL_TIMEOUT_SECONDS,
    OLLAMA_HOST,
    PERSONAL_COLUMN_WORDS,
)
from smart_data_studio.dataset import is_sensitive

PROMPT = """Which of these columns hold personal data about an individual?

Reply with a JSON array of column names and nothing else. [] if none do.

You are shown names and types and no values, deliberately — this runs before the
file is read, so that asking cannot itself disclose anything.

Include a column when it names, contacts or locates a person, records something
about their body or life, or identifies them: names, emails, phone numbers, dates
of birth, home addresses, postcodes, government or account numbers held by a
person, and anything that would identify somebody once set beside another column
in this list.

Exclude business measures, event dates, categories, and identifiers of things
rather than people — a machine, a game, an order, a store. A player or customer
id is a person's identifier and belongs in the list."""


def by_name(columns: list[str]) -> set[str]:
    """The columns a fixed word list calls personal. The floor under any proposal."""
    return {
        name
        for name in columns
        if is_sensitive(name) or any(word in name.lower() for word in PERSONAL_COLUMN_WORDS)
    }


def propose(schema: dict[str, list[tuple[str, str]]]) -> set[str]:
    """Column names worth withholding, from the model where it answers and from the
    word list either way.

    Never raises. A model that is down, slow or talking nonsense must not stop a
    file loading — it makes the proposal worse, and the floor still holds.
    """
    columns = sorted({name for table in schema.values() for name, _ in table})
    found = by_name(columns)
    listed = "\n".join(
        f"{table}: " + ", ".join(f"{name} ({kind})" for name, kind in table_columns)
        for table, table_columns in schema.items()
    )
    try:
        reply = ollama.Client(host=OLLAMA_HOST, timeout=MODEL_TIMEOUT_SECONDS).chat(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": listed},
            ],
        )
        named = _names_in(reply["message"]["content"])
    except Exception as error:
        logs.event("sensitive.proposal.failed", reason=str(error)[:120])
        return found
    # Only names this data actually has: a model naming a column that is not here
    # would otherwise withhold nothing while looking like it had.
    known = {name.lower(): name for name in columns}
    proposed = {known[name.lower()] for name in named if name.lower() in known}
    logs.event("sensitive.proposed", proposed=len(proposed), by_name=len(found))
    return found | proposed


def _names_in(text: str) -> list[str]:
    """The JSON array of strings in a reply, whatever surrounds it."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        found = json.loads(text[start : end + 1])
    except ValueError:
        return []
    return [item for item in found if isinstance(item, str)] if isinstance(found, list) else []
