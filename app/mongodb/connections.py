import asyncio
import logging

from .profiles import _normalize_app_name, update_user_profile

logger = logging.getLogger(__name__)


async def sync_composio_connections(user_id: str, composio_instance=None) -> dict[str, bool] | None:
    """Refresh active integrations from Composio and cache them in the profile."""
    if composio_instance is None:
        from ..utils.composio_config import composio
        composio_instance = composio

    connected_apps: dict[str, bool] = {}
    try:
        accounts = await asyncio.to_thread(
            composio_instance.connected_accounts.list,
            user_ids=[user_id],
            statuses=["ACTIVE"],
        )
        for account in getattr(accounts, "items", []) or []:
            app_name = getattr(account, "toolkit", None)
            if not app_name:
                app_name = getattr(account, "app_unique_id", None) or getattr(account, "app_name", None)
            if app_name:
                connected_apps[_normalize_app_name(app_name)] = True
    except Exception as error:
        logger.warning("Could not fetch connected accounts from Composio for %s: %s", user_id, error)
        return None

    await update_user_profile(user_id, {"connected_apps": connected_apps})
    return connected_apps
