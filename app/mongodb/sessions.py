import logging
from datetime import datetime, timezone
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from .client import ensure_ttl_index, get_db

logger = logging.getLogger(__name__)

MAX_TURNS = 20
TOOL_TRACE_MAX_ENTRIES = 8


def _tool_trace_note(trace: Any) -> str | None:
    """Render stored tool activity as a compact note for context replay."""
    if not isinstance(trace, list):
        return None
    entries: list[str] = []
    for item in trace[:TOOL_TRACE_MAX_ENTRIES]:
        if isinstance(item, dict):
            name = str(item.get("tool") or "tool").strip()
            result = str(item.get("result") or "").strip()
            entries.append(f"{name} ({result})" if result else name)
        elif str(item).strip():
            entries.append(str(item).strip())
    return "[Tool activity in your last turn: " + "; ".join(entries) + "]" if entries else None


def _pairs_to_model_messages(pairs: list[Any]) -> list[ModelMessage]:
    """Rebuild pydantic-ai history from saved user/assistant pairs."""
    messages: list[ModelMessage] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        user_message = pair.get("user_message")
        if not isinstance(user_message, str) or not user_message:
            continue
        agent_message = pair.get("agent_response")
        agent_message = agent_message if isinstance(agent_message, str) else ""
        note = _tool_trace_note(pair.get("tool_trace"))
        if note:
            agent_message = f"{agent_message}\n{note}".strip()
        messages.append(ModelRequest(parts=[UserPromptPart(content=user_message)]))
        if agent_message:
            messages.append(ModelResponse(parts=[TextPart(content=agent_message)]))
    return messages


def _legacy_history_to_pairs(stored: list[Any]) -> list[dict[str, str]]:
    """Flatten old full pydantic-ai dumps into compact user/assistant pairs."""
    try:
        messages = ModelMessagesTypeAdapter.validate_python(stored)
    except Exception as error:
        logger.warning("Could not parse legacy session history: %s", error)
        return []
    pairs: list[dict[str, str]] = []
    user_message: str | None = None
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    user_message = part.content
        elif isinstance(message, ModelResponse) and user_message is not None:
            text = "".join(part.content for part in message.parts if isinstance(part, TextPart)).strip()
            if text:
                pairs.append({"user_message": user_message, "agent_response": text})
                user_message = None
    return pairs


async def load_session_history(user_id: str, session_id: str) -> list[ModelMessage]:
    """Load compact history and migrate legacy dumps on first read."""
    db = get_db()
    if db is None:
        return []
    session_key = f"{user_id}:{session_id}"
    try:
        doc = await db.chat_sessions.find_one({"_id": session_key})
        stored: list[Any] = doc.get("messages", []) if doc else []
        is_legacy = bool(stored and isinstance(stored[0], dict) and "parts" in stored[0])
        pairs = _legacy_history_to_pairs(stored) if is_legacy else stored
        compact_pairs = pairs[-MAX_TURNS:]

        # Drop prompts, reasoning, tool payloads, and provider metadata in old dumps.
        if (is_legacy and pairs) or len(pairs) > MAX_TURNS:
            try:
                await db.chat_sessions.update_one(
                    {"_id": session_key},
                    {"$set": {"messages": compact_pairs}},
                )
            except Exception as error:
                logger.warning("Could not compact session history for %s: %s", session_key, error)

        return _pairs_to_model_messages(compact_pairs)
    except Exception as error:
        logger.error("Error loading session history for %s: %s", session_key, error)
        return []


async def save_session_history(
    user_id: str,
    session_id: str,
    user_message: str,
    agent_message: str,
    tool_trace: list[dict[str, str]] | None = None,
) -> None:
    """Append a turn, retaining at most the latest 20 pairs."""
    db = get_db()
    if db is None:
        return
    await ensure_ttl_index()
    session_key = f"{user_id}:{session_id}"
    try:
        update: dict[str, Any] = {
            "$set": {
                "user_id": user_id,
                "session_id": session_id,
                "updated_at": datetime.now(timezone.utc),
            }
        }
        if agent_message:
            pair: dict[str, Any] = {"user_message": user_message, "agent_response": agent_message}
            if tool_trace:
                pair["tool_trace"] = tool_trace
            update["$push"] = {"messages": {"$each": [pair], "$slice": -MAX_TURNS}}
        await db.chat_sessions.update_one({"_id": session_key}, update, upsert=True)
    except Exception as error:
        logger.error("Error saving session history for %s: %s", session_key, error)


async def delete_session_history(session_id: str, user_id: str | None = None) -> None:
    """Delete session history from MongoDB."""
    db = get_db()
    if db is None:
        return
    try:
        if user_id:
            await db.chat_sessions.delete_one({"_id": f"{user_id}:{session_id}"})
        else:
            await db.chat_sessions.delete_many({"session_id": session_id})
    except Exception as error:
        logger.error("Error deleting session %s: %s", session_id, error)
