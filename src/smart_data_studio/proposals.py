"""What the model proposed about how tables relate, resolved against the schema.

A proposal is a hypothesis. Every reference is resolved against the loaded schema
before any query is built, and one naming something withheld, unknown or malformed
is rejected rather than repaired.

Metadata is keyed by table *and* column, never by column alone — two files
commonly hold the same column name meaning different things.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_data_studio.config import (
    MAX_JOIN_CANDIDATES,
    MAX_KEY_CANDIDATES,
    MAX_KEY_COLUMNS,
)
from smart_data_studio.dataset import Dataset


@dataclass(frozen=True)
class Ref:
    """One side of a candidate: a table and an ordered tuple of its columns."""

    table: str
    columns: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.table}({', '.join(self.columns)})"


# reason is annotation, not identity: the same join proposed twice with different
# wording is one candidate.
@dataclass(frozen=True)
class KeyCandidate:
    ref: Ref
    reason: str = field(default="", compare=False)


@dataclass(frozen=True)
class JoinCandidate:
    left: Ref
    right: Ref
    reason: str = field(default="", compare=False)

    def __str__(self) -> str:
        return f"{self.left} = {self.right}"


@dataclass
class Proposals:
    keys: list[KeyCandidate] = field(default_factory=list)
    joins: list[JoinCandidate] = field(default_factory=list)
    # Why each rejected proposal was refused. A silent drop would look identical to
    # the model never having proposed anything.
    rejected: list[str] = field(default_factory=list)


class Invalid(ValueError):
    """A proposal that cannot be resolved against the loaded data."""


def _resolve(dataset: Dataset, table: str, columns: list[str] | tuple[str, ...]) -> Ref:
    """Match a proposal against the schema, returning the schema's own spelling.

    Compared without case so `playerid` resolves, but what is stored is what the
    file calls the column — anything else produces SQL that does not run.
    """
    known = {name.lower(): name for name in dataset.tables}
    actual_table = known.get(str(table).strip().lower())
    if actual_table is None:
        raise Invalid(f"unknown table {table!r}")

    schema = {name.lower(): name for name, _ in dataset.schema(actual_table)}
    if not columns:
        raise Invalid(f"{actual_table}: no columns given")
    if len(columns) > MAX_KEY_COLUMNS:
        raise Invalid(f"{actual_table}: {len(columns)} columns, more than {MAX_KEY_COLUMNS}")

    resolved: list[str] = []
    for column in columns:
        name = schema.get(str(column).strip().lower())
        if name is None:
            raise Invalid(f"{actual_table} has no column {column!r}")
        if name in resolved:
            raise Invalid(f"{actual_table}: {name} repeated")
        resolved.append(name)
    return Ref(actual_table, tuple(resolved))


def validate(dataset: Dataset, raw: list[dict]) -> Proposals:
    """Turn whatever the model proposed into candidates that resolve, or reasons.

    Bounded: every candidate costs a verification query.
    """
    found = Proposals()
    per_table: dict[str, int] = {}
    per_pair: dict[tuple[str, str], int] = {}

    for item in raw:
        try:
            kind = str(item.get("kind", "")).strip().lower()
            if kind == "key":
                ref = _resolve(dataset, item.get("table", ""), item.get("columns") or [])
                if per_table.get(ref.table, 0) >= MAX_KEY_CANDIDATES:
                    raise Invalid(f"{ref.table}: more than {MAX_KEY_CANDIDATES} key candidates")
                candidate = KeyCandidate(ref, str(item.get("reason", "")).strip())
                if candidate not in found.keys:
                    per_table[ref.table] = per_table.get(ref.table, 0) + 1
                    found.keys.append(candidate)
            elif kind == "join":
                left_raw, right_raw = item.get("left") or {}, item.get("right") or {}
                left = _resolve(dataset, left_raw.get("table", ""), left_raw.get("columns") or [])
                right = _resolve(
                    dataset, right_raw.get("table", ""), right_raw.get("columns") or []
                )
                if left.table == right.table:
                    raise Invalid(f"{left.table} joined to itself")
                if len(left.columns) != len(right.columns):
                    raise Invalid(f"{left} and {right} have different numbers of columns")
                pair = tuple(sorted((left.table, right.table)))
                if per_pair.get(pair, 0) >= MAX_JOIN_CANDIDATES:
                    raise Invalid(
                        f"{pair[0]}/{pair[1]}: more than {MAX_JOIN_CANDIDATES} join candidates"
                    )
                candidate = JoinCandidate(left, right, str(item.get("reason", "")).strip())
                if candidate not in found.joins:
                    per_pair[pair] = per_pair.get(pair, 0) + 1
                    found.joins.append(candidate)
            else:
                raise Invalid(f"unknown proposal kind {item.get('kind')!r}")
        except Invalid as error:
            found.rejected.append(str(error))
        except (AttributeError, TypeError) as error:
            found.rejected.append(f"malformed proposal: {type(error).__name__}: {error}")
    return found
