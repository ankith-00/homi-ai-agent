from pydantic import BaseModel

class ChatIn(BaseModel):
    session_id: str
    user_id: str
    message: str