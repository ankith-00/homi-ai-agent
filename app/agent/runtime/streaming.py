"""Streaming execution loop and public agent API."""

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic_ai.agent import CallToolsNode, ModelRequestNode
from pydantic_ai.exceptions import (
    ModelConnectionError,
    ModelHTTPError,
    ModelQuotaExceededError,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    TextPartDelta,
)

from ...mongodb import save_session_history
from ...observability import CAPTURE_CHAT_CONTENT, content_for_trace, span
from ...utils.composio_config import composio_mcp
from ...utils.mem0_config import add_memory
from .config import MODEL_LABEL, agent
from .errors import (
    AgentBusyError,
    FallbackExceptionGroup,
    TRANSIENT_MODEL_ERRORS,
    format_model_error_details,
)
from .instrumentation import log_pending_operation, tool_trace
from .session import prepare_session

logger = logging.getLogger(__name__)
model_logger = logging.getLogger("app.model")

MODEL_RETRY_ATTEMPTS = 2
MODEL_RETRY_DELAY_SECONDS = 2.0
MODEL_STREAM_IDLE_TIMEOUT_SECONDS = 45.0  # cancel if no token arrives within this window


@dataclass
class AgentStreamChunk:
    delta: str | None = None
    final: str | None = None
    tool: dict[str, str] | None = None


async def _stream_agent_impl(
    session_id: str,
    user_id: str,
    text: str,
    turn_span: Any,
    turn_id: str,
) -> AsyncIterator[AgentStreamChunk]:
    preparation_started = perf_counter()
    session = await prepare_session(session_id, user_id, text, turn_id)
    preparation_ms = round((perf_counter() - preparation_started) * 1000)
    turn_span.set_attribute("homi.preparation_ms", preparation_ms)
    streamed_any = False
    first_text_seen = False

    logger.info(
        "Agent turn started turn=%s model=%s input=%s",
        turn_id,
        MODEL_LABEL,
        json.dumps(content_for_trace(text), ensure_ascii=False),
    )

    for attempt in range(MODEL_RETRY_ATTEMPTS + 1):
        tool_call_count = 0
        tool_started_at: dict[str, float] = {}
        try:
            toolset = composio_mcp(user_id)
            logger.info("Opening agent run turn=%s attempt=%d toolset=Composio-MCP", turn_id, attempt + 1)
            agent_run_started = perf_counter()
            async with agent.iter(
                text,
                deps=session,
                message_history=session.history,
                toolsets=[toolset],
            ) as agent_run:
                logger.info(
                    "Agent run initialized turn=%s attempt=%d setup_ms=%d preparation_ms=%d",
                    turn_id,
                    attempt + 1,
                    round((perf_counter() - agent_run_started) * 1000),
                    preparation_ms,
                )
                model_round = 0
                async for node in agent_run:
                    if isinstance(node, ModelRequestNode):
                        model_round += 1
                        model_started_at = perf_counter()
                        async with log_pending_operation(turn_id, f"model_request_round_{model_round}") as mark_model_activity:
                            try:
                                request_messages = [*agent_run.all_messages(), node.request]
                                model_payload = ModelMessagesTypeAdapter.dump_json(request_messages).decode("utf-8")
                                logger.info(
                                    "Model request payload turn=%s round=%d messages=%s",
                                    turn_id,
                                    model_round,
                                    json.dumps(content_for_trace(model_payload), ensure_ascii=False),
                                )
                            except Exception:
                                logger.exception("Could not serialize model request payload turn=%s round=%d", turn_id, model_round)

                            target_model = MODEL_LABEL
                            model_logger.info(
                                "--> [MODEL DISPATCH] Calling model='%s' | Round: %d | Attempt: %d/%d",
                                target_model,
                                model_round,
                                attempt + 1,
                                MODEL_RETRY_ATTEMPTS + 1,
                            )
                            logger.info("Model request sent turn=%s round=%d", turn_id, model_round)
                            async with node.stream(agent_run.ctx) as stream:
                                mark_model_activity()
                                connect_ms = round((perf_counter() - model_started_at) * 1000)
                                model_logger.info(
                                    "<-- [MODEL CONNECTED] Stream opened | Model: '%s' | Connect time: %dms",
                                    target_model,
                                    connect_ms,
                                )
                                logger.info(
                                    "Model response stream opened turn=%s round=%d elapsed_ms=%d",
                                    turn_id,
                                    model_round,
                                    connect_ms,
                                )
                                stream_iter = stream.__aiter__()
                                while True:
                                    try:
                                        event = await asyncio.wait_for(
                                            stream_iter.__anext__(),
                                            timeout=MODEL_STREAM_IDLE_TIMEOUT_SECONDS,
                                        )
                                    except StopAsyncIteration:
                                        break
                                    except asyncio.TimeoutError:
                                        model_logger.error(
                                            "[MODEL ERROR] Stream stalled (no tokens for %ds) | Model: '%s'",
                                            int(MODEL_STREAM_IDLE_TIMEOUT_SECONDS),
                                            target_model,
                                        )
                                        logger.error(
                                            "Model stream idle timeout turn=%s round=%d idle_limit_s=%s",
                                            turn_id,
                                            model_round,
                                            MODEL_STREAM_IDLE_TIMEOUT_SECONDS,
                                        )
                                        raise ModelConnectionError(
                                            f"Model stream stalled – no response from '{target_model}' within {int(MODEL_STREAM_IDLE_TIMEOUT_SECONDS)}s"
                                        )
                                    mark_model_activity()
                                    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                        if event.delta.content_delta:
                                            streamed_any = True
                                            logger.info(
                                                "Model text delta turn=%s round=%d text=%s",
                                                turn_id,
                                                model_round,
                                                json.dumps(content_for_trace(event.delta.content_delta), ensure_ascii=False),
                                            )
                                            if not first_text_seen:
                                                first_text_seen = True
                                                first_text_ms = round((perf_counter() - preparation_started) * 1000)
                                                model_logger.info(
                                                    "<-- [MODEL TOKEN] First token received | Model: '%s' | TTFT: %dms",
                                                    target_model,
                                                    first_text_ms,
                                                )
                                                turn_span.set_attribute(
                                                    "homi.time_to_first_text_ms",
                                                    first_text_ms,
                                                )
                                                logger.info("Agent first text turn=%s elapsed_ms=%d", turn_id, first_text_ms)
                                            yield AgentStreamChunk(delta=event.delta.content_delta)
                        round_ms = round((perf_counter() - model_started_at) * 1000)
                        model_logger.info(
                            "<-- [MODEL DONE] Round %d completed | Model: '%s' | Total time: %dms",
                            model_round,
                            target_model,
                            round_ms,
                        )
                        logger.info(
                            "Model request finished turn=%s round=%d duration_ms=%d",
                            turn_id,
                            model_round,
                            round_ms,
                        )
                    elif isinstance(node, CallToolsNode):
                        async with log_pending_operation(turn_id, f"tool_execution_round_{model_round}") as mark_tool_activity:
                            async with node.stream(agent_run.ctx) as stream:
                                async for event in stream:
                                    mark_tool_activity()
                                    if isinstance(event, FunctionToolCallEvent):
                                        tool_call_count += 1
                                        tool_name = event.part.tool_name
                                        tool_started_at[event.part.tool_call_id] = perf_counter()
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
                                        if result_content is None:
                                            result_content = getattr(event.part, "content", None)
                                        result_summary = content_for_trace(str(result_content)) if result_content is not None else "<no content>"
                                        call_started_at = tool_started_at.pop(event.part.tool_call_id, None)
                                        tool_duration_ms = (
                                            round((perf_counter() - call_started_at) * 1000)
                                            if call_started_at is not None
                                            else None
                                        )
                                        logger.info(
                                            "Tool call finished turn=%s tool=%s duration_ms=%s result=%s",
                                            turn_id,
                                            event.part.tool_name,
                                            tool_duration_ms,
                                            result_summary,
                                        )
                                        yield AgentStreamChunk(
                                            tool={
                                                "tool": event.part.tool_name,
                                                "status": "done",
                                            }
                                        )

                output = agent_run.result.output if agent_run.result else ""
                tool_trace_entries = tool_trace(agent_run.new_messages())

            with span("homi.session.save"):
                await save_session_history(user_id, session_id, text, output, tool_trace_entries)
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
        except ModelHTTPError as error:
            err_details = format_model_error_details(error)
            model_logger.error(
                "[MODEL ERROR] Provider HTTP Error | Model: '%s' | %s",
                MODEL_LABEL,
                err_details,
            )
            logger.exception("Model HTTP error turn=%s attempt=%d", turn_id, attempt + 1)
            raise AgentBusyError(
                f"Model error ({getattr(error, 'status_code', 'HTTP failure')}). Check your model name and provider status."
            ) from error
        except TRANSIENT_MODEL_ERRORS as error:
            err_details = format_model_error_details(error)
            model_logger.error(
                "[MODEL ERROR] Transient Failure (Attempt %d/%d) | Model: '%s' | %s",
                attempt + 1,
                MODEL_RETRY_ATTEMPTS + 1,
                MODEL_LABEL,
                err_details,
            )
            logger.exception("Transient model error turn=%s attempt=%d", turn_id, attempt + 1)
            if streamed_any or tool_call_count > 0 or attempt >= MODEL_RETRY_ATTEMPTS:
                if tool_call_count > 0:
                    logger.error(
                        "Not retrying turn=%s attempt=%d because %d tool call(s) already ran",
                        turn_id,
                        attempt + 1,
                        tool_call_count,
                    )
                    raise AgentBusyError(
                        "The request was interrupted after an action may have been performed. "
                        "Please verify the result before trying again."
                    ) from error
                raise AgentBusyError(
                    "Homi Agent is busy right now. Please try again in a few seconds."
                ) from error
            retry_wait = min(max(getattr(error, "retry_after", None) or MODEL_RETRY_DELAY_SECONDS, 0.5), 5.0)
            model_logger.warning("Retrying model in %.1fs...", retry_wait)
            await asyncio.sleep(retry_wait)
        except FallbackExceptionGroup as error:
            err_details = format_model_error_details(error)
            model_logger.error(
                "[MODEL ERROR] All fallback models failed | Chain: '%s' | %s",
                MODEL_LABEL,
                err_details,
            )
            logger.exception("All fallback models failed turn=%s attempt=%d", turn_id, attempt + 1)
            if tool_call_count > 0:
                logger.error(
                    "Not retrying turn=%s attempt=%d because %d tool call(s) already ran",
                    turn_id,
                    attempt + 1,
                    tool_call_count,
                )
                raise AgentBusyError(
                    "The request was interrupted after an action may have been performed. "
                    "Please verify the result before trying again."
                ) from error
            raise AgentBusyError(
                "Homi Agent is busy right now. Please try again in a few seconds."
            ) from error
        except ModelQuotaExceededError as error:
            err_details = format_model_error_details(error)
            model_logger.error(
                "[MODEL ERROR] Provider Quota Exceeded | Model: '%s' | %s",
                MODEL_LABEL,
                err_details,
            )
            logger.exception("Model quota error turn=%s", turn_id)
            raise AgentBusyError(
                "Homi Agent has hit its provider quota. Please try again later."
            ) from error
        except Exception as error:
            err_details = format_model_error_details(error)
            model_logger.error(
                "[MODEL ERROR] Unexpected model exception | Model: '%s' | %s",
                MODEL_LABEL,
                err_details,
            )
            raise


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
