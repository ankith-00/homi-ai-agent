import os
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ModelConnectionError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelServerError,
    ModelUnavailableError,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..utils.composio_config import composio_mcp
from ..utils.mem0_config import add_memory, search_memories
from ..utils.mongo_config import (
    get_or_create_user_profile,
    sync_composio_connections,
    load_session_history,
    save_session_history,
)
from .components.context import Session
from .components.prompts import inject_user_context
from .components.tools import AGENT_TOOLS

load_dotenv()

MAX_MESSAGES = 30

MODEL_RETRY_ATTEMPTS = 1
MODEL_RETRY_DELAY_SECONDS = 2.0

TRANSIENT_MODEL_ERRORS = (
    ModelRateLimitError,
    ModelUnavailableError,
    ModelServerError,
    ModelConnectionError,
)


class AgentBusyError(Exception):
    """Raised when the model provider is temporarily unable to serve a request."""


sessions: dict[str, Session] = {}


# Model Configuration
model = OpenAIChatModel(
    os.getenv("NVIDIA_MODEL"),
    provider=OpenAIProvider(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    ),
)

INSTRUCTIONS = (Path(__file__).parent / "instructions.md").read_text(encoding="utf-8")

# Agent Configuration
agent = Agent(
    model,
    deps_type=Session,
    system_prompt=INSTRUCTIONS,
)

for tool_function in AGENT_TOOLS:
    agent.tool(tool_function)
agent.system_prompt(inject_user_context)
def trim(history: list[ModelMessage]) -> list[ModelMessage]:
    history = history[-MAX_MESSAGES:]
    while history and not (
        isinstance(history[0], ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in history[0].parts)
    ):
        history = history[1:]
    return history




@dataclass
class AgentStreamChunk:
    delta: str | None = None
    final: str | None = None


async def _prepare_session(session_id: str, user_id: str, text: str) -> Session:
    history, profile, relevant_memories = await asyncio.gather(
        load_session_history(user_id, session_id),
        get_or_create_user_profile(user_id),
        search_memories(query=text, user_id=user_id, top_k=5),
    )

    connected_apps = await sync_composio_connections(user_id)
    if connected_apps is not None:
        profile["connected_apps"] = connected_apps

    return Session(
        user_id=user_id,
        profile=profile,
        history=history,
        relevant_memories=relevant_memories,
    )


async def stream_agent(session_id: str, user_id: str, text: str) -> AsyncIterator[AgentStreamChunk]:
    session = await _prepare_session(session_id, user_id, text)
    streamed_any = False

    for attempt in range(MODEL_RETRY_ATTEMPTS + 1):
        try:
            async with agent.run_stream(
                text,
                deps=session,
                message_history=session.history,
                toolsets=[composio_mcp(user_id)],
            ) as result:
                async for delta in result.stream_text(delta=True):
                    if delta:
                        streamed_any = True
                        yield AgentStreamChunk(delta=delta)
                output = await result.get_output()
                updated_history = trim(result.all_messages())

            await save_session_history(user_id, session_id, updated_history)
            asyncio.create_task(
                add_memory(
                    [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": output},
                    ],
                    user_id=user_id,
                )
            )
            yield AgentStreamChunk(final=output)
            return
        except TRANSIENT_MODEL_ERRORS as error:
            if streamed_any or attempt >= MODEL_RETRY_ATTEMPTS:
                raise AgentBusyError(
                    "Homi Agent is busy right now. Please try again in a few seconds."
                ) from error
            await asyncio.sleep(min(max(error.retry_after or MODEL_RETRY_DELAY_SECONDS, 0.5), 5.0))
        except ModelQuotaExceededError as error:
            raise AgentBusyError(
                "Homi Agent has hit its provider quota. Please try again later."
            ) from error


async def run_agent(session_id: str, user_id: str, text: str) -> str:
    async for chunk in stream_agent(session_id, user_id, text):
        if chunk.final is not None:
            return chunk.final
    raise AgentBusyError("Homi Agent did not produce a reply.")
