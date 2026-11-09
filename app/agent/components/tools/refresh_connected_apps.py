from pydantic_ai import RunContext

from ....mongodb import sync_composio_connections
from ..context import Session


async def refresh_connected_apps(ctx: RunContext[Session]) -> str:
    """Refresh the user's active integrations from Composio and cache them."""
    connected = await sync_composio_connections(ctx.deps.user_id)
    if connected is None:
        return "Could not reach Composio to refresh connected apps. Your saved apps are unchanged."
    ctx.deps.profile["connected_apps"] = connected

    active_apps = [app for app, is_active in connected.items() if is_active]
    if active_apps:
        return f"Connected apps updated: {', '.join(active_apps)}"
    return "No active integrations found."
