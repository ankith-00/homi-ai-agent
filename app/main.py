import base64

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.agent.core import run_agent, sessions
from utils.elevenlabs_config import stt, tts

app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True}

class ChatIn(BaseModel):
    session_id: str
    user_id: str
    message: str

@app.post("/chat")
async def chat(body: ChatIn):
    return {"reply": await run_agent(body.session_id, body.user_id, body.message)}



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
async def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}
