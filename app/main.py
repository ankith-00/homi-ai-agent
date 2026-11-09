import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure application logging level. Set LOG_LEVEL in .env (e.g. WARNING or ERROR to silence logs).
log_level_name = os.getenv("LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, log_level_name, logging.WARNING)

app_logger = logging.getLogger("app")
app_logger.setLevel(log_level)
app_logger.propagate = False
if not any(getattr(handler, "_homi_agent_console", False) for handler in app_logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(name)s: %(message)s"))
    console_handler._homi_agent_console = True
    app_logger.addHandler(console_handler)

# Keep model status, latency, and errors always visible in the terminal
logging.getLogger("app.model").setLevel(logging.INFO)

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
