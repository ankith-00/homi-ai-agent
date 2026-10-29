"""MongoDB access, grouped by the data each module owns."""

from .client import ensure_ttl_index, get_db
from .connections import sync_composio_connections
from .profiles import get_or_create_user_profile, get_user_profile, update_user_profile
from .sessions import delete_session_history, load_session_history, save_session_history

__all__ = [
    "delete_session_history",
    "ensure_ttl_index",
    "get_db",
    "get_or_create_user_profile",
    "get_user_profile",
    "load_session_history",
    "save_session_history",
    "sync_composio_connections",
    "update_user_profile",
]
