from fastapi import APIRouter, HTTPException

from app.utils.mongo_config import get_user_profile

router = APIRouter(tags=["users"])


@router.get("/user/{user_id}/")
async def user_profile(user_id: str):
    profile = await get_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="User profile not found")
    return {"profile": profile}
