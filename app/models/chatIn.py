from pydantic import BaseModel

class ChatIn(BaseModel):
    session_id: str
    user_id: str
    audio_out: bool
    message: str
