"""Structured logging: one JSON object per event, never a raw data value.

An operator has to be able to diagnose a failed question — which query, how long,
how many rows, which model — without being handed the user's CSV. So events carry
shapes, timings and identifiers, and the SQL itself is the one payload allowed
through, because it is already shown to the user beside every answer.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from smart_data_studio.config import LOG_FORMAT, LOG_LEVEL, PROMPT_VERSION, VERSION

# Correlation flows through context rather than call signatures, so a tool buried
# in the agent loop still reports which session and question it belongs to.
_session: ContextVar[str] = ContextVar("session", default="-")
_question: ContextVar[str] = ContextVar("question", default="-")

_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": record.getMessage(),
            "session": _session.get(),
            "question": _question.get(),
            "version": VERSION,
            "prompt_version": PROMPT_VERSION,
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(payload, default=str)


def configure() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _JsonFormatter()
        if LOG_FORMAT == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    root = logging.getLogger("smart_data_studio")
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)
    root.propagate = False
    _configured = True


def new_session() -> str:
    """An opaque id. Nothing about the user or their file goes into it."""
    return uuid.uuid4().hex[:12]


def bind(session: str | None = None, question: str | None = None) -> None:
    if session is not None:
        _session.set(session)
    if question is not None:
        _question.set(question)


def event(name: str, **fields: object) -> None:
    configure()
    logging.getLogger("smart_data_studio").info(name, extra={"fields": fields})


def failure(name: str, **fields: object) -> None:
    configure()
    logging.getLogger("smart_data_studio").exception(name, extra={"fields": fields})


@contextmanager
def timed(name: str, **fields: object):
    """Emit one event with a duration, whether the block succeeds or raises."""
    started = time.perf_counter()
    try:
        yield fields
    except Exception:
        fields["ms"] = round((time.perf_counter() - started) * 1000, 1)
        failure(f"{name}.failed", **fields)
        raise
    fields["ms"] = round((time.perf_counter() - started) * 1000, 1)
    event(name, **fields)
