from fastapi import APIRouter

from app.utils.mem0_config import get_mem0_client
from app.utils.mongo_config import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    status = {"ok": True, "services": {}}

    try:
        db = get_db()
        await db.command("ping")
        status["services"]["mongodb"] = "connected"
    except Exception as e:
        status["services"]["mongodb"] = f"error: {e}"
        status["ok"] = False

    try:
        mem0 = get_mem0_client()
        if mem0:
            status["services"]["mem0"] = "connected"
        else:
            status["services"]["mem0"] = "error: client not initialized"
            status["ok"] = False
    except Exception as e:
        status["services"]["mem0"] = f"error: {e}"
        status["ok"] = False

    return status
