import os
import logging
from time import perf_counter
from composio import Composio
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPToolset

load_dotenv()

logger = logging.getLogger(__name__)

AVAILABLE_APPS = {
    "gmail": "Gmail",
    "googlecalendar": "Google Calendar",
    "notion": "Notion",
    "googledocs": "Google Docs",
    "github": "GitHub",
}
TOOLKITS = list(AVAILABLE_APPS)

composio = Composio(api_key=os.environ["COMPOSIO_API_KEY"])
cache: dict[str, MCPToolset] = {}


def composio_mcp(user_id: str) -> MCPToolset:
    if user_id not in cache:
        logger.info("Creating Composio MCP session user_hash=%s toolkits=%s", _user_hash(user_id), TOOLKITS)
        started_at = perf_counter()
        try:
            session = composio.create(user_id=user_id, toolkits=TOOLKITS)
            cache[user_id] = MCPToolset(session.mcp.url, headers=session.mcp.headers)
            logger.info(
                "Composio MCP session ready user_hash=%s duration_ms=%d",
                _user_hash(user_id),
                round((perf_counter() - started_at) * 1000),
            )
        except Exception:
            logger.exception("Composio MCP session creation failed user_hash=%s", _user_hash(user_id))
            raise
    else:
        logger.info("Reusing cached Composio MCP session user_hash=%s", _user_hash(user_id))
    return cache[user_id]


def _user_hash(user_id: str) -> str:
    import hashlib

    return hashlib.sha256(user_id.encode()).hexdigest()[:12]
