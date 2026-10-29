from fastapi import APIRouter

from app.agent.core import sessions
from app.mongodb import delete_session_history

router = APIRouter(tags=["sessions"])


@router.delete("/session/{session_id}")
async def reset(session_id: str, user_id: str | None = None):
    await delete_session_history(session_id, user_id)

    if user_id:
        sessions.pop(f"{user_id}:{session_id}", None)
    else:
        keys_to_del = [
            key for key in sessions
            if key.endswith(f":{session_id}") or key == session_id
        ]
        for key in keys_to_del:
            sessions.pop(key, None)

    return {"ok": True}
