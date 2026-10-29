from fastapi import APIRouter

from app.utils.mem0_config import get_mem0_client
from app.mongodb import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    status = {"ok": True, "Author": "Ankith Kumar", "Portfolio": "https://ankith.is-a.dev"}
    return status
