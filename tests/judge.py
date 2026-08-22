"""Grading the prose of an answer, which no anchor can reach.

Both banks anchor on values proved separately in SQL, so they catch a wrong
number. They cannot catch a right number described wrongly — a share whose
denominator was never established, a cause asserted that the data cannot show —
because nothing in them reads the sentences. A second model does that here,
against a fixed rubric.

Two rules make the verdicts worth having. A finding must quote the words it
objects to, because a judge returning a bare boolean is unfalsifiable and one
that has to point at a sentence is far less inclined to invent a fault. And the
judge is calibrated before it is trusted: planted answers whose faults are known
go through it first, so a judge that cannot tell a clean answer from a broken one
fails loudly rather than quietly grading real work.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import ollama

from smart_data_studio.config import MODEL_ID, OLLAMA_HOST

# A different model from the one under test would be better — a model grading its
# own habits forgives them. One config value, so an endpoint that serves two can
# use two without touching the code.
JUDGE_MODEL_ID = os.environ.get("SDS_JUDGE_MODEL_ID", MODEL_ID)

# Every fault the judge looks for, and the words it is given to look for it with.
# Each is a mode we have watched happen, not a category invented for symmetry.
RUBRIC: dict[str, str] = {
    "unsupported_figure": (
        "A number, ranking or superlative that appears in the answer but in none of "
        "the query results below. Quoting a figure the queries did not produce. Only "
        "when the evidence reproduces the whole result — where it says rows were "
        "withheld, a figure you cannot find is not thereby unsupported."
    ),
    "invented_cause": (
        "Asserting WHY something happened by reaching for something the data does "
        "not contain — a price change, a campaign, a competitor, a payout, weather, "
        "an intention. Saying a figure is 'likely driven by' or 'explained by' a "
        "mechanism no query could show.\n"
        "    NOT this fault: arithmetic between figures that are present. If the "
        "evidence gives visits and players, then 'they visit more often' is a "
        "division, not a cause; if it gives two totals, their ratio is not a cause. "
        "An analyst deriving one number from two is doing the job, and the "
        "derivation does not have to appear in the evidence for the claim to rest "
        "on it. Only fault a mechanism that no arithmetic over the evidence reaches. "
        "The word 'because' does not decide it: what decides it is whether the thing "
        "named is in the evidence."
    ),
    "unstated_base": (
        "A share, rate, percentage or 'X of Y' whose denominator is never stated or "
        "is not the one the question asks about. A retention or conversion figure "
        "given without saying what population it is a share of."
    ),
    "overstated_coverage": (
        "Describing a truncated, sampled or filtered result as though it covered "
        "everything. Calling a partial result 'all', 'every' or 'the total' when a "
        "query returned only some rows, or omitting that a result was truncated."
    ),
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": sorted(RUBRIC)},
                    "failed": {"type": "boolean"},
                    "quote": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["dimension", "failed", "quote", "why"],
            },
        }
    },
    "required": ["findings"],
}

PROMPT = """You are grading one data analyst's answer against the evidence behind it.

You are not judging whether the answer is useful, well written or thorough. You are
checking four specific faults and nothing else. Report each fault separately.

For every fault, decide whether the answer commits it. If it does, quote the exact
words from the answer that commit it — verbatim, copied, not paraphrased. If it does
not, set failed to false and leave the quote empty.

Be conservative. A hedge is not a cause: "which may reflect seasonality, though this
data cannot show that" states its own limit and passes. A figure the answer explicitly
attributes to a query passes. Silence is not a fault — an answer that simply does not
mention something is not committing anything. Only mark a fault you can quote.

The faults:
{rubric}

Reply with JSON and nothing else — no prose, no YAML, no commentary — in exactly
this shape, one entry per fault and four entries in total:

{{"findings": [
  {{"dimension": "unsupported_figure", "failed": false, "quote": "", "why": ""}},
  {{"dimension": "invented_cause", "failed": true,
   "quote": "copied verbatim from the answer", "why": "one short line"}},
  {{"dimension": "unstated_base", "failed": false, "quote": "", "why": ""}},
  {{"dimension": "overstated_coverage", "failed": false, "quote": "", "why": ""}}
]}}"""


def _first_json_object(text: str) -> dict:
    """The JSON object in a reply, whatever prose or fencing surrounds it.

    `format` is passed on every call and ignored by some endpoints — the hosted
    one this was built against returns a fenced block regardless — so the reply is
    parsed rather than trusted, the same way agent._first_json_list does.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        found = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


class JudgeUnusable(RuntimeError):
    """The judge returned nothing usable, which is not the same as a clean answer."""


@dataclass(frozen=True)
class Finding:
    dimension: str
    failed: bool
    quote: str
    why: str


def evidence_of(answer, sample_rows: int = 50, max_chars: int = 14_000) -> str:
    """The queries behind an answer, as the judge sees them.

    Generous on purpose. An analytical result is usually an aggregate of a few
    dozen rows, and showing three of twenty-five taught the judge to call a figure
    unsupported when it was merely on a row it had not been given — the harness
    manufacturing the fault it was built to find. Where rows genuinely are withheld
    the evidence says so, so absence can be read as absence of proof rather than
    proof of absence.
    """
    blocks = []
    for index, result in enumerate(answer.results, start=1):
        frame = result.frame.head(sample_rows)
        withheld = len(result.frame) > len(frame) or result.truncated
        blocks.append(
            f"Query {index}:\n{result.sql}\n"
            f"Returned {result.total_rows:,} rows. "
            + (
                f"Only {len(frame):,} of them are reproduced here, so a figure you cannot "
                f"find below may still be in the result — do not call it unsupported on "
                f"that basis alone:\n"
                if withheld
                else f"All {len(frame):,} are reproduced here:\n"
            )
            + frame.to_string(index=False)
        )
    for record in answer.analyses:
        blocks.append(f"Analysis ({record.kind}) of {record.subject}:\n{json.dumps(record.result)}")
    if not blocks:
        return "No query was run and no analysis was computed."
    joined = "\n\n".join(blocks)
    return joined[:max_chars] + ("\n… (evidence truncated)" if len(joined) > max_chars else "")


def grade(question: str, answer_text: str, evidence: str, client=None) -> dict[str, Finding]:
    """Run one answer past the rubric, keyed by dimension."""
    rubric = "\n".join(f"- {name}: {description}" for name, description in RUBRIC.items())
    reply = (client or ollama.Client(host=OLLAMA_HOST)).chat(
        model=JUDGE_MODEL_ID,
        format=_SCHEMA,
        messages=[
            {"role": "system", "content": PROMPT.format(rubric=rubric)},
            {
                "role": "user",
                "content": (
                    f"QUESTION\n{question}\n\n"
                    f"ANSWER\n{answer_text}\n\n"
                    f"EVIDENCE (every query that ran)\n{evidence}"
                ),
            },
        ],
    )
    text = reply.message.content or ""
    found = _first_json_object(text).get("findings", [])
    graded = {
        item["dimension"]: Finding(
            dimension=item["dimension"],
            failed=bool(item.get("failed")),
            quote=str(item.get("quote", "")),
            why=str(item.get("why", "")),
        )
        for item in found
        if isinstance(item, dict) and item.get("dimension") in RUBRIC
    }
    # A dimension the judge did not rule on is not a clean one. Padding it to
    # "not failed" is how an unparseable reply reads as four passes — which is
    # precisely what happened the first time this ran and the endpoint answered
    # in YAML. There is no silent pass here: either every fault was ruled on or
    # the judge is unusable for this answer and says so.
    missing = sorted(set(RUBRIC) - set(graded))
    if missing:
        raise JudgeUnusable(
            f"no verdict for {', '.join(missing)}. The judge replied: {text[:400]!r}"
        )
    return graded


def quotes(quote: str, answer_text: str) -> bool:
    """Whether the quote was copied from the answer rather than paraphrased."""
    squash = re.compile(r"\s+")
    if not quote.strip():
        return False
    return squash.sub(" ", quote).strip().lower() in squash.sub(" ", answer_text).strip().lower()


def failures(graded: dict[str, Finding], answer_text: str) -> list[Finding]:
    """The faults the judge found and could point at in the answer.

    A fault it cannot quote is neither counted nor dropped. Dropping it is a
    silent pass and counting it trusts a judge that may have invented the whole
    thing, so it raises: what it means is that this verdict needs a human, not
    that the answer was clean.
    """
    found = [item for item in graded.values() if item.failed]
    unquotable = [item for item in found if not quotes(item.quote, answer_text)]
    if unquotable:
        raise JudgeUnusable(
            "found a fault it could not quote from the answer: "
            + "; ".join(f"{item.dimension} said {item.quote[:90]!r}" for item in unquotable)
        )
    return found


def describe(graded: dict[str, Finding], answer_text: str) -> str:
    """A one-line summary for a failure message.

    Takes the answer for the same reason failures() does, and never raises: this
    runs while a test is already failing, and a describe() that throws replaces
    the finding you need to read with a TypeError.
    """
    found = [item for item in graded.values() if item.failed]
    if not found:
        return "clean"
    return "; ".join(
        f"{item.dimension}: {item.quote[:120]!r}"
        + ("" if quotes(item.quote, answer_text) else " (NOT QUOTED FROM THE ANSWER)")
        for item in found
    )
