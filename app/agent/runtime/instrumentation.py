"""Logging and tracing helpers for staged work and tool calls."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from time import perf_counter

from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart

logger = logging.getLogger(__name__)

TOOL_TRACE_MAX_ENTRIES = 8


@asynccontextmanager
async def log_pending_operation(turn_id: str, stage: str, interval_seconds: int = 30):
    """Log an agent stage if it stops producing events for the interval."""
    started_at = perf_counter()
    last_activity_at = started_at

    def mark_activity() -> None:
        nonlocal last_activity_at
        last_activity_at = perf_counter()

    async def report_wait() -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            idle_ms = round((perf_counter() - last_activity_at) * 1000)
            if idle_ms >= interval_seconds * 1000:
                logger.warning(
                    "Agent stage has no events turn=%s stage=%s idle_ms=%d elapsed_ms=%d",
                    turn_id,
                    stage,
                    idle_ms,
                    round((perf_counter() - started_at) * 1000),
                )

    watcher = asyncio.create_task(report_wait())
    try:
        yield mark_activity
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


def tool_result_digest(tool_name: str) -> str:
    """Return a safe, one-line summary without inspecting tool payloads."""
    if tool_name.upper().endswith("MANAGE_CONNECTIONS"):
        return "connection link created"
    return "completed"


def tool_trace(messages: list[ModelMessage]) -> list[dict[str, str]]:
    """Digest this run's tool calls/returns into compact trace entries."""
    trace: list[dict[str, str]] = []
    by_call_id: dict[str, dict[str, str]] = {}
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                entry = {"tool": part.tool_name}
                by_call_id[part.tool_call_id] = entry
                trace.append(entry)
            elif isinstance(part, ToolReturnPart):
                entry = by_call_id.get(part.tool_call_id)
                if entry is None:
                    entry = {"tool": part.tool_name}
                    trace.append(entry)
                if "result" not in entry:
                    entry["result"] = tool_result_digest(part.tool_name)
    return trace[-TOOL_TRACE_MAX_ENTRIES:]
