from fastapi import APIRouter, HTTPException

from app.utils.composio_config import composio
from app.mongodb import get_user_profile, sync_composio_connections

router = APIRouter(tags=["users"])


@router.get("/user/{user_id}/")
async def user_profile(user_id: str):
    profile = await get_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="User profile not found")
    connected = await sync_composio_connections(user_id, composio)
    if connected is not None:
        profile["connected_apps"] = connected
    return {"profile": profile}
