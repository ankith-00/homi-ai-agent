import os
import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from time import perf_counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai import RunContext
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

logger = logging.getLogger(__name__)

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
logger.info(
    "Agent instructions loaded chars=%d content=%s",
    len(INSTRUCTIONS),
    json.dumps(content_for_trace(INSTRUCTIONS), ensure_ascii=False),
)

def build_agent_system_prompt(ctx: RunContext[Session]) -> str:
    """Render the static instruction template and per-user context per turn."""
    profile = ctx.deps.profile
    custom_attrs = profile.get("custom_attributes", {})
    preferences = profile.get("preferences", {})
    timezone_name = (
        profile.get("timezone")
        or (preferences.get("timezone") if isinstance(preferences, dict) else None)
        or (custom_attrs.get("timezone") if isinstance(custom_attrs, dict) else None)
        or "UTC"
    )
    if not isinstance(timezone_name, str):
        timezone_name = "UTC"
    try:
        user_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            "Unknown profile timezone turn=%s timezone=%s; using UTC",
            ctx.deps.turn_id,
            timezone_name,
        )
        timezone_name = "UTC"
        user_timezone = ZoneInfo("UTC")

    user_name = " ".join(str(profile.get("name") or "the user").split())[:80]
    now = datetime.now(user_timezone)
    rendered_instructions = (
        INSTRUCTIONS.replace("{{user_name}}", str(user_name))
        .replace("{{current_datetime}}", now.strftime("%Y-%m-%d %H:%M:%S %Z"))
        .replace("{{timezone}}", timezone_name)
    )
    prompt = rendered_instructions + inject_user_context(ctx)
    logger.info(
        "System prompt composed turn=%s chars=%d timezone=%s content=%s",
        ctx.deps.turn_id,
        len(prompt),
        timezone_name,
        json.dumps(content_for_trace(prompt), ensure_ascii=False),
    )
    return prompt


# Agent Configuration
agent = Agent(model, deps_type=Session)

for tool_function in AGENT_TOOLS:
    agent.tool(tool_function)
agent.system_prompt(dynamic=True)(build_agent_system_prompt)
configure_observability(agent)
logger.info(
    "Agent startup ready instructions=loaded instruction_chars=%d model=%s registered_tools=%s composio_mcp=configured_per_user_on_first_request",
    len(INSTRUCTIONS),
    os.getenv("NVIDIA_MODEL", "<unset>"),
    [tool_function.__name__ for tool_function in AGENT_TOOLS],
)


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


async def _prepare_session(session_id: str, user_id: str, text: str, turn_id: str) -> Session:
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
        "Agent context ready user=%s session=%s history_messages=%d memories=%d connected_apps=%s",
        hashlib.sha256(user_id.encode()).hexdigest()[:12],
        hashlib.sha256(session_id.encode()).hexdigest()[:12],
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


async def _stream_agent_impl(
    session_id: str,
    user_id: str,
    text: str,
    turn_span: Any,
    turn_id: str,
) -> AsyncIterator[AgentStreamChunk]:
    preparation_started = perf_counter()
    session = await _prepare_session(session_id, user_id, text, turn_id)
    turn_span.set_attribute("homi.preparation_ms", round((perf_counter() - preparation_started) * 1000))
    streamed_any = False
    first_text_seen = False
    tool_call_count = 0

    logger.info(
        "Agent turn started turn=%s model=%s input=%s",
        turn_id,
        os.getenv("NVIDIA_MODEL", "<unset>"),
        json.dumps(content_for_trace(text), ensure_ascii=False),
    )

    for attempt in range(MODEL_RETRY_ATTEMPTS + 1):
        try:
            toolset = composio_mcp(user_id)
            logger.info("Opening agent run turn=%s attempt=%d toolset=Composio-MCP", turn_id, attempt + 1)
            async with agent.iter(
                text,
                deps=session,
                message_history=session.history,
                toolsets=[toolset],
            ) as agent_run:
                model_round = 0
                async for node in agent_run:
                    if isinstance(node, ModelRequestNode):
                        model_round += 1
                        logger.info("Model request turn=%s round=%d", turn_id, model_round)
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
                                    tool_name = event.part.tool_name
                                    arguments = event.part.args
                                    try:
                                        safe_arguments = content_for_trace(json.dumps(arguments, ensure_ascii=False, default=str))
                                    except Exception:
                                        safe_arguments = "<unavailable>"
                                    logger.info(
                                        "Tool call started turn=%s tool=%s args=%s",
                                        turn_id,
                                        tool_name,
                                        safe_arguments,
                                    )
                                    yield AgentStreamChunk(tool={"tool": tool_name, "status": "started"})
                                elif isinstance(event, FunctionToolResultEvent):
                                    result_content = getattr(event, "content", None)
                                    result_summary = content_for_trace(str(result_content)) if result_content is not None else "<no content>"
                                    logger.info(
                                        "Tool call finished turn=%s tool=%s result=%s",
                                        turn_id,
                                        event.part.tool_name,
                                        result_summary,
                                    )
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
            logger.info(
                "Agent turn completed turn=%s tool_calls=%d output=%s",
                turn_id,
                tool_call_count,
                json.dumps(content_for_trace(output), ensure_ascii=False),
            )
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
            logger.exception("Transient model error turn=%s attempt=%d", turn_id, attempt + 1)
            if streamed_any or attempt >= MODEL_RETRY_ATTEMPTS:
                raise AgentBusyError(
                    "Homi Agent is busy right now. Please try again in a few seconds."
                ) from error
            await asyncio.sleep(min(max(error.retry_after or MODEL_RETRY_DELAY_SECONDS, 0.5), 5.0))
        except ModelQuotaExceededError as error:
            logger.exception("Model quota error turn=%s", turn_id)
            raise AgentBusyError(
                "Homi Agent has hit its provider quota. Please try again later."
            ) from error


async def stream_agent(
    session_id: str,
    user_id: str,
    text: str,
    *,
    turn_id: str | None = None,
) -> AsyncIterator[AgentStreamChunk]:
    """Trace one chat turn and stream its model/tool events to the caller."""
    turn_id = turn_id or str(uuid.uuid4())
    attributes: dict[str, Any] = {
        "homi.turn_id": turn_id,
        "homi.user_hash": hashlib.sha256(user_id.encode()).hexdigest()[:16],
        "homi.session_hash": hashlib.sha256(session_id.encode()).hexdigest()[:16],
        "homi.input_length": len(text),
        "homi.content_capture_enabled": CAPTURE_CHAT_CONTENT,
    }
    if CAPTURE_CHAT_CONTENT:
        attributes["homi.user_message"] = content_for_trace(text)

    with span("homi.chat.turn", **attributes) as turn_span:
        try:
            async for chunk in _stream_agent_impl(session_id, user_id, text, turn_span, turn_id):
                yield chunk
        except Exception:
            logger.exception("Agent turn failed turn=%s", turn_id)
            raise


async def run_agent(session_id: str, user_id: str, text: str, *, turn_id: str | None = None) -> str:
    async for chunk in stream_agent(session_id, user_id, text, turn_id=turn_id):
        if chunk.final is not None:
            return chunk.final
    raise AgentBusyError("Homi Agent did not produce a reply.")
