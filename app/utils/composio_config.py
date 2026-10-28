import os
from composio import Composio
from dotenv import load_dotenv
from pydantic_ai.mcp import MCPToolset

load_dotenv()

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
        session = composio.create(user_id=user_id, toolkits=TOOLKITS)
        cache[user_id] = MCPToolset(session.mcp.url, headers=session.mcp.headers)
    return cache[user_id]
