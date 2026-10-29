import os
import asyncio
import hashlib
import uuid
from time import perf_counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.agent import ModelRequestNode, CallToolsNode
from pydantic_ai.exceptions import (
    ModelConnectionError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelServerError,
    ModelUnavailableError,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..utils.composio_config import composio_mcp
from ..utils.mem0_config import add_memory, search_memories
from ..observability import CAPTURE_CHAT_CONTENT, configure_observability, content_for_trace, span
from ..mongodb import (
    get_or_create_user_profile,
    load_session_history,
    save_session_history,
)
from .components.context import Session
from .components.prompts import inject_user_context
from .components.tools import AGENT_TOOLS

load_dotenv()

MODEL_RETRY_ATTEMPTS = 1
MODEL_RETRY_DELAY_SECONDS = 2.0

TOOL_TRACE_MAX_ENTRIES = 8

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
configure_observability(agent)


def _tool_result_digest(tool_name: str) -> str:
    """Return a safe, one-line summary without inspecting tool payloads."""
    if tool_name.upper().endswith("MANAGE_CONNECTIONS"):
        return "connection link created"
    return "completed"


def _tool_trace(messages: list[ModelMessage]) -> list[dict[str, str]]:
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
                    entry["result"] = _tool_result_digest(part.tool_name)
    return trace[-TOOL_TRACE_MAX_ENTRIES:]




@dataclass
class AgentStreamChunk:
    delta: str | None = None
    final: str | None = None
    tool: dict[str, str] | None = None


async def _prepare_session(session_id: str, user_id: str, text: str) -> Session:
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

    return Session(
        user_id=user_id,
        profile=profile,
        history=history,
        relevant_memories=relevant_memories,
    )


async def _stream_agent_impl(
    session_id: str,
    user_id: str,
    text: str,
    turn_span: Any,
) -> AsyncIterator[AgentStreamChunk]:
    preparation_started = perf_counter()
    session = await _prepare_session(session_id, user_id, text)
    turn_span.set_attribute("homi.preparation_ms", round((perf_counter() - preparation_started) * 1000))
    streamed_any = False
    first_text_seen = False
    tool_call_count = 0

    for attempt in range(MODEL_RETRY_ATTEMPTS + 1):
        try:
            async with agent.iter(
                text,
                deps=session,
                message_history=session.history,
                toolsets=[composio_mcp(user_id)],
            ) as agent_run:
                async for node in agent_run:
                    if isinstance(node, ModelRequestNode):
                        async with node.stream(agent_run.ctx) as stream:
                            async for event in stream:
                                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                    if event.delta.content_delta:
                                        streamed_any = True
                                        if not first_text_seen:
                                            first_text_seen = True
                                            turn_span.set_attribute(
                                                "homi.time_to_first_text_ms",
                                                round((perf_counter() - preparation_started) * 1000),
                                            )
                                        yield AgentStreamChunk(delta=event.delta.content_delta)
                    elif isinstance(node, CallToolsNode):
                        async with node.stream(agent_run.ctx) as stream:
                            async for event in stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    tool_call_count += 1
                                    yield AgentStreamChunk(tool={"tool": event.call.tool_name, "status": "started"})
                                elif isinstance(event, FunctionToolResultEvent):
                                    yield AgentStreamChunk(
                                        tool={
                                            "tool": event.part.tool_name,
                                            "status": "done",
                                        }
                                    )

                output = agent_run.result.output if agent_run.result else ""
                tool_trace = _tool_trace(agent_run.new_messages())

            with span("homi.session.save"):
                await save_session_history(user_id, session_id, text, output, tool_trace)
            turn_span.set_attribute("homi.tool_call_count", tool_call_count)
            turn_span.set_attribute("homi.output_length", len(output))
            turn_span.set_attribute("homi.status", "completed")
            if CAPTURE_CHAT_CONTENT:
                turn_span.set_attribute("homi.final_response", content_for_trace(output))
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


async def stream_agent(session_id: str, user_id: str, text: str) -> AsyncIterator[AgentStreamChunk]:
    """Trace one chat turn and stream its model/tool events to the caller."""
    attributes: dict[str, Any] = {
        "homi.turn_id": str(uuid.uuid4()),
        "homi.user_hash": hashlib.sha256(user_id.encode()).hexdigest()[:16],
        "homi.session_hash": hashlib.sha256(session_id.encode()).hexdigest()[:16],
        "homi.input_length": len(text),
        "homi.content_capture_enabled": CAPTURE_CHAT_CONTENT,
    }
    if CAPTURE_CHAT_CONTENT:
        attributes["homi.user_message"] = content_for_trace(text)

    with span("homi.chat.turn", **attributes) as turn_span:
        async for chunk in _stream_agent_impl(session_id, user_id, text, turn_span):
            yield chunk


async def run_agent(session_id: str, user_id: str, text: str) -> str:
    async for chunk in stream_agent(session_id, user_id, text):
        if chunk.final is not None:
            return chunk.final
    raise AgentBusyError("Homi Agent did not produce a reply.")
