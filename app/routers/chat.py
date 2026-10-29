import asyncio
import base64
import hashlib
import json
import logging
import re
import uuid
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agent.core import AgentBusyError, AgentStreamChunk, run_agent, stream_agent
from app.models.chatIn import ChatIn
from app.observability import content_for_trace
from app.utils.elevenlabs_config import tts

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)
TRACE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9-]{8,80}$")


def _request_trace_id(request: Request) -> str:
    supplied = request.headers.get("x-homi-trace-id", "").strip()
    return supplied if TRACE_ID_PATTERN.fullmatch(supplied) else str(uuid.uuid4())


def _log_received(turn_id: str, body: ChatIn, message: str, connect_request: str) -> None:
    logger.info(
        "Chat request received turn=%s user=%s session=%s message=%s connect_request=%s audio_out=%s",
        turn_id,
        hashlib.sha256(body.user_id.strip().encode()).hexdigest()[:12],
        hashlib.sha256(body.session_id.strip().encode()).hexdigest()[:12],
        json.dumps(content_for_trace(body.message), ensure_ascii=False),
        json.dumps(content_for_trace(connect_request), ensure_ascii=False),
        body.audio_out,
    )
    if message != body.message.strip():
        logger.info("Agent input after request normalization turn=%s message=%s", turn_id, json.dumps(content_for_trace(message), ensure_ascii=False))


@router.post("/chat")
async def chat(body: ChatIn, request: Request):
    turn_id = _request_trace_id(request)
    user_id = body.user_id.strip()
    session_id = body.session_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    message = body.message.strip()
    connect_request = (body.connect_request or "").strip()

    # Frontend integration command format: /connect <app name or slug>
    if message.startswith("/"):
        command, _, argument = message[1:].partition(" ")
        if command.strip().lower() == "connect":
            connect_request = connect_request or argument.strip() or "an integration"
            message = ""

    if connect_request:
        connection_prompt = (
            "The user requested to connect an integration."
            if connect_request == "an integration"
            else f"Connect request from the user: {connect_request}"
        )
        message = f"{message}\n\n{connection_prompt}" if message else connection_prompt
    if not message:
        raise HTTPException(status_code=400, detail="message or connect_request cannot be empty")

    _log_received(turn_id, body, message, connect_request)

    agent_started = perf_counter()
    logger.info("Agent phase started turn=%s audio_out=%s", turn_id, body.audio_out)
    try:
        agent_response = await run_agent(session_id, user_id, message, turn_id=turn_id)
    except AgentBusyError as error:
        logger.warning("Chat request unavailable turn=%s error=%s", turn_id, error)
        raise HTTPException(status_code=503, detail=str(error))
    logger.info(
        "Agent phase finished turn=%s duration_ms=%d",
        turn_id,
        round((perf_counter() - agent_started) * 1000),
    )

    if body.audio_out:
        tts_started = perf_counter()
        logger.info("TTS started turn=%s", turn_id)
        audio_bytes = await tts(agent_response)
        logger.info(
            "TTS finished turn=%s duration_ms=%d audio_bytes=%d",
            turn_id,
            round((perf_counter() - tts_started) * 1000),
            len(audio_bytes),
        )
        # The voice client consumes JSON with the spoken reply bundled in, so it
        # can caption while the audio plays.
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {"reply": agent_response, "audio": audio_b64}

    return {"reply": agent_response}


@router.post("/chat/stream")
async def chat_stream(body: ChatIn, request: Request):
    turn_id = _request_trace_id(request)
    user_id = body.user_id.strip()
    session_id = body.session_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")
    if body.audio_out:
        raise HTTPException(status_code=400, detail="streaming chat does not support audio output; use /chat")

    message = body.message.strip()
    connect_request = (body.connect_request or "").strip()

    if message.startswith("/"):
        command, _, argument = message[1:].partition(" ")
        if command.strip().lower() == "connect":
            connect_request = connect_request or argument.strip() or "an integration"
            message = ""

    if connect_request:
        connection_prompt = (
            "The user requested to connect an integration."
            if connect_request == "an integration"
            else f"Connect request from the user: {connect_request}"
        )
        message = f"{message}\n\n{connection_prompt}" if message else connection_prompt
    if not message:
        raise HTTPException(status_code=400, detail="message or connect_request cannot be empty")

    _log_received(turn_id, body, message, connect_request)

    async def event_stream():
        events: asyncio.Queue[tuple[str, AgentStreamChunk | Exception | None]] = asyncio.Queue(maxsize=1)

        async def produce_agent_events() -> None:
            agent_events = stream_agent(session_id, user_id, message, turn_id=turn_id)
            cancelled = False
            failure: Exception | None = None
            try:
                async for chunk in agent_events:
                    await events.put(("chunk", chunk))
            except asyncio.CancelledError:
                cancelled = True
            except Exception as error:
                failure = error
            finally:
                # Close the async generator in this producer task so its
                # Logfire/OpenTelemetry span detaches in the context it attached.
                try:
                    await agent_events.aclose()
                except Exception as close_error:
                    if failure is None:
                        failure = close_error

            if cancelled:
                raise asyncio.CancelledError
            if failure is not None:
                await events.put(("error", failure))
            await events.put(("end", None))

        producer = asyncio.create_task(produce_agent_events())
        try:
            while True:
                try:
                    event_type, payload = await asyncio.wait_for(events.get(), timeout=15)
                except asyncio.TimeoutError:
                    # Keep the Next.js proxy and hosting layer from treating a
                    # long-running tool call as a dead stream. SSE clients ignore comments.
                    yield ": keep-alive\n\n"
                    continue

                if event_type == "end":
                    return
                if event_type == "error":
                    error = payload
                    if isinstance(error, AgentBusyError):
                        logger.warning("Streaming chat unavailable turn=%s error=%s", turn_id, error)
                        yield f"event: error\ndata: {json.dumps({'message': str(error)})}\n\n"
                    elif isinstance(error, Exception):
                        logger.error(
                            "Streaming chat failed turn=%s",
                            turn_id,
                            exc_info=(type(error), error, error.__traceback__),
                        )
                        yield f"event: error\ndata: {json.dumps({'message': 'Something went wrong. Please try again.'})}\n\n"
                    else:
                        logger.error("Streaming chat failed turn=%s with an unknown error", turn_id)
                        yield f"event: error\ndata: {json.dumps({'message': 'Something went wrong. Please try again.'})}\n\n"
                    continue

                if not isinstance(payload, AgentStreamChunk):
                    continue
                chunk = payload
                if chunk.delta is not None:
                    yield f"event: delta\ndata: {json.dumps({'text': chunk.delta})}\n\n"
                elif chunk.tool is not None:
                    yield f"event: tool\ndata: {json.dumps(chunk.tool)}\n\n"
                elif chunk.final is not None:
                    yield f"event: done\ndata: {json.dumps({'reply': chunk.final})}\n\n"
        finally:
            if not producer.done():
                producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
