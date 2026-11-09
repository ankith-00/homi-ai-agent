import base64
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.core import AgentBusyError, run_agent, stream_agent
from app.models.chatIn import ChatIn
from app.utils.elevenlabs_config import tts

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(body: ChatIn):
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

    try:
        agent_response = await run_agent(session_id, user_id, message)
    except AgentBusyError as error:
        raise HTTPException(status_code=503, detail=str(error))

    if body.audio_out:
        audio_bytes = await tts(agent_response)
        # The voice client consumes JSON with the spoken reply bundled in, so it
        # can caption while the audio plays.
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {"reply": agent_response, "audio": audio_b64}

    return {"reply": agent_response}


@router.post("/chat/stream")
async def chat_stream(body: ChatIn):
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

    async def event_stream():
        try:
            async for chunk in stream_agent(session_id, user_id, message):
                if chunk.delta is not None:
                    yield f"event: delta\ndata: {json.dumps({'text': chunk.delta})}\n\n"
                elif chunk.final is not None:
                    yield f"event: done\ndata: {json.dumps({'reply': chunk.final})}\n\n"
        except AgentBusyError as error:
            yield f"event: error\ndata: {json.dumps({'message': str(error)})}\n\n"
        except Exception:
            yield f"event: error\ndata: {json.dumps({'message': 'Something went wrong. Please try again.'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
