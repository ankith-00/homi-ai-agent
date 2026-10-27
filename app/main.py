import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response

from app.agent.core import run_agent, sessions
from app.models.chatIn import ChatIn
from app.utils.elevenlabs_config import tts
from app.utils.mongo_config import delete_session_history

app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True}



@app.post("/chat")
async def chat(body: ChatIn):
    user_id = body.user_id.strip()
    session_id = body.session_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    return {"reply": await run_agent(session_id, user_id, body.message)}



# @app.post("/voice")
# async def voice(
#     session_id: str = Form(...),
#     user_id: str = Form(...),
#     audio: UploadFile = File(...),
# ):
#     text = await stt(await audio.read(), audio.filename or "audio.webm")
#     reply = await run_agent(session_id, user_id, text)
#     return Response(await tts(reply), media_type="audio/mpeg")

@app.delete("/session/{session_id}")
async def reset(session_id: str, user_id: str | None = None):
    # 1. Delete from MongoDB persistent store
    await delete_session_history(session_id, user_id)
    
    # 2. Delete from in-memory fallback if present
    if user_id:
        sessions.pop(f"{user_id}:{session_id}", None)
    else:
        keys_to_del = [k for k in sessions if k.endswith(f":{session_id}") or k == session_id]
        for k in keys_to_del:
            sessions.pop(k, None)
    return {"ok": True}
