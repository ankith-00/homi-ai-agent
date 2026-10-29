import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.chat import router as chat_router
from app.routers.health import router as health_router
from app.routers.sessions import router as sessions_router
from app.routers.users import router as users_router
from app.utils.auth import api_key_auth

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://project-voi.vercel.app"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

app.include_router(chat_router, dependencies=[Depends(api_key_auth)])
app.include_router(health_router)
app.include_router(users_router, dependencies=[Depends(api_key_auth)])
app.include_router(sessions_router, dependencies=[Depends(api_key_auth)])
