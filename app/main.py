import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI

from app.routers.chat import router as chat_router
from app.routers.health import router as health_router
from app.routers.sessions import router as sessions_router
from app.routers.users import router as users_router

app = FastAPI()

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(users_router)
app.include_router(sessions_router)