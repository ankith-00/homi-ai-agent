"""Per-turn session preparation and the in-memory session registry."""

import asyncio
import hashlib
import logging
from time import perf_counter

from ...mongodb import get_or_create_user_profile, load_session_history
from ...observability import span
from ...utils.mem0_config import search_memories
from ..components.context import Session

logger = logging.getLogger(__name__)

sessions: dict[str, Session] = {}


async def prepare_session(session_id: str, user_id: str, text: str, turn_id: str) -> Session:
    preparation_started = perf_counter()
    with span("homi.session.prepare") as preparation_span:

        async def load_history():
            with span("homi.session.load_history"):
                return await load_session_history(user_id, session_id)

        async def load_profile():
            with span("homi.profile.load"):
                return await get_or_create_user_profile(user_id)

        async def find_memories():
            with span("homi.memory.search"):
                return await search_memories(query=text, user_id=user_id, top_k=5)

        history, profile, relevant_memories = await asyncio.gather(
            load_history(),
            load_profile(),
            find_memories(),
        )
        preparation_span.set_attribute("homi.history_message_count", len(history))
        preparation_span.set_attribute("homi.memory_result_count", len(relevant_memories))

    logger.info(
        "Agent context ready turn=%s user=%s session=%s duration_ms=%d history_messages=%d memories=%d connected_apps=%s",
        turn_id,
        hashlib.sha256(user_id.encode()).hexdigest()[:12],
        hashlib.sha256(session_id.encode()).hexdigest()[:12],
        round((perf_counter() - preparation_started) * 1000),
        len(history),
        len(relevant_memories),
        sorted(app for app, active in profile.get("connected_apps", {}).items() if active),
    )

    return Session(
        user_id=user_id,
        turn_id=turn_id,
        profile=profile,
        history=history,
        relevant_memories=relevant_memories,
    )
