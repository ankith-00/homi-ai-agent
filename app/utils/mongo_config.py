import os
import logging
import re
import asyncio
from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

load_dotenv()

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(86400 * 2)))  # 48 hours default

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None
_ttl_initialized = False


def get_db() -> AsyncIOMotorDatabase | None:
    """Lazily initializes and returns the async MongoDB database instance."""
    global _client, _db
    if _db is not None:
        return _db

    if not MONGODB_URI:
        logger.warning("MONGODB_URI not found in environment variables.")
        return None

    try:
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client[DB_NAME]
        return _db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None


async def ensure_ttl_index():
    """Ensure TTL index exists on chat_sessions so old chats auto-expire."""
    global _ttl_initialized
    if _ttl_initialized:
        return
    db = get_db()
    if db is None:
        return
    try:
        await db.chat_sessions.create_index("updated_at", expireAfterSeconds=SESSION_TTL_SECONDS)
        _ttl_initialized = True
    except Exception as e:
        logger.warning(f"Could not initialize TTL index for chat_sessions: {e}")


async def load_session_history(user_id: str, session_id: str) -> list[ModelMessage]:
    """Load conversation history for a given user and session from MongoDB."""
    db = get_db()
    if db is None:
        return []
    session_key = f"{user_id}:{session_id}"
    try:
        doc = await db.chat_sessions.find_one({"_id": session_key})
        if doc and "messages" in doc:
            return ModelMessagesTypeAdapter.validate_python(doc["messages"])
    except Exception as e:
        logger.error(f"Error loading session history for {session_key}: {e}")
    return []


async def save_session_history(user_id: str, session_id: str, messages: list[ModelMessage]):
    """Persist conversation history to MongoDB with auto-expiring TTL."""
    db = get_db()
    if db is None:
        return
    await ensure_ttl_index()
    session_key = f"{user_id}:{session_id}"
    try:
        serialized = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        await db.chat_sessions.update_one(
            {"_id": session_key},
            {
                "$set": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": serialized,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
    except Exception as e:
        logger.error(f"Error saving session history for {session_key}: {e}")


async def delete_session_history(session_id: str, user_id: str | None = None):
    """Delete session history from MongoDB."""
    db = get_db()
    if db is None:
        return
    try:
        if user_id:
            await db.chat_sessions.delete_one({"_id": f"{user_id}:{session_id}"})
        else:
            await db.chat_sessions.delete_many({"session_id": session_id})
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")


def _normalize_app_name(value: Any) -> str:
    """Convert Composio toolkit values to clean app keys such as ``gmail``."""
    slug = getattr(value, "slug", None)
    if slug:
        value = slug

    name = str(value).strip().lower()
    match = re.fullmatch(r"slug\s*=\s*['\"]([^'\"]+)['\"]", name)
    if match:
        return match.group(1).strip().lower()
    return name


def _normalize_connected_apps(apps: dict[str, bool]) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    for app_name, is_connected in apps.items():
        clean_name = _normalize_app_name(app_name)
        if clean_name:
            normalized[clean_name] = normalized.get(clean_name, False) or is_connected
    return normalized


async def get_user_profile(user_id: str) -> dict | None:
    """Retrieve user profile document from MongoDB by user_id."""
    db = get_db()
    if db is None:
        return None
    try:
        profile = await db.user_profiles.find_one({"_id": user_id})
        if profile and isinstance(profile.get("connected_apps"), dict):
            connected_apps = _normalize_connected_apps(profile["connected_apps"])
            if connected_apps != profile["connected_apps"]:
                await db.user_profiles.update_one(
                    {"_id": user_id},
                    {"$set": {"connected_apps": connected_apps}},
                )
                profile["connected_apps"] = connected_apps
        return profile
    except Exception as e:
        logger.error(f"Error fetching profile for {user_id}: {e}")
        return None


async def get_or_create_user_profile(user_id: str) -> dict:
    """
    Get existing user profile or initialize a new one in MongoDB.
    """
    db = get_db()
    if db is None:
        return {
            "_id": user_id,
            "name": None,
            "email": None,
            "age": None,
            "location": None,
            "connected_apps": {},
            "preferences": {},
            "custom_attributes": {},
        }

    try:
        profile = await db.user_profiles.find_one({"_id": user_id})
        if not profile:
            now = datetime.now(timezone.utc)
            new_profile = {
                "_id": user_id,
                "name": None,
                "email": None,
                "age": None,
                "location": None,
                "connected_apps": {},
                "preferences": {},
                "custom_attributes": {},
                "created_at": now,
                "updated_at": now,
            }
            await db.user_profiles.insert_one(new_profile)
            return new_profile
        return profile
    except Exception as e:
        logger.error(f"Error in get_or_create_user_profile for {user_id}: {e}")
        return {
            "_id": user_id,
            "name": None,
            "email": None,
            "age": None,
            "location": None,
            "connected_apps": {},
            "preferences": {},
            "custom_attributes": {},
        }


async def update_user_profile(user_id: str, updates: dict[str, Any]) -> dict | None:
    """
    Update specific fields in user profile (e.g. name, preferences, connected_apps).
    """
    db = get_db()
    if db is None:
        return None

    updates["updated_at"] = datetime.now(timezone.utc)
    try:
        return await db.user_profiles.find_one_and_update(
            {"_id": user_id},
            {"$set": updates},
            upsert=True,
            return_document=True,
        )
    except Exception as e:
        logger.error(f"Error updating profile for {user_id}: {e}")
        return None


async def sync_composio_connections(user_id: str, composio_instance=None) -> dict[str, bool] | None:
    """
    Sync active Composio integrations for this user and cache them in MongoDB.

    Returns the active app map, or None when Composio could not be reached
    (in which case the cached value in MongoDB is left untouched).
    """
    if composio_instance is None:
        from .composio_config import composio
        composio_instance = composio

    connected_apps: dict[str, bool] = {}

    try:
        # Check active connected accounts in Composio (SDK call is synchronous;
        # run it in a worker thread so the async event loop stays responsive)
        accounts = await asyncio.to_thread(
            composio_instance.connected_accounts.list,
            user_ids=[user_id],
            statuses=["ACTIVE"],
        )
        items = getattr(accounts, "items", []) or []
        for acc in items:
            app_name = getattr(acc, "toolkit", None)
            if not app_name:
                app_name = getattr(acc, "app_unique_id", None) or getattr(acc, "app_name", None)
            if app_name:
                connected_apps[_normalize_app_name(app_name)] = True
    except Exception as e:
        logger.warning(f"Could not fetch connected accounts from Composio for {user_id}: {e}")
        return None

    # Cache the active apps in MongoDB (also when empty, so disconnects propagate)
    await update_user_profile(user_id, {"connected_apps": connected_apps})

    return connected_apps
