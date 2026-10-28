from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.agent.core import run_agent
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

    agent_response = await run_agent(session_id, user_id, message)

    if body.audio_out:
        audio_bytes = await tts(agent_response)
        return Response(content=audio_bytes, media_type="audio/mpeg")

    return {"reply": agent_response}
