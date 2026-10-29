import logging
import os
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .client import get_db

logger = logging.getLogger(__name__)
PROFILE_CACHE_TTL_SECONDS = max(0, int(os.getenv("USER_PROFILE_CACHE_TTL_SECONDS", "60")))
_profile_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _get_cached_profile(user_id: str) -> dict[str, Any] | None:
    cached = _profile_cache.get(user_id)
    if cached is None:
        return None
    cached_at, profile = cached
    if time.monotonic() - cached_at >= PROFILE_CACHE_TTL_SECONDS:
        _profile_cache.pop(user_id, None)
        return None
    return deepcopy(profile)


def _cache_profile(user_id: str, profile: dict[str, Any]) -> None:
    if PROFILE_CACHE_TTL_SECONDS > 0:
        _profile_cache[user_id] = (time.monotonic(), deepcopy(profile))


def _normalize_app_name(value: Any) -> str:
    """Convert Composio toolkit values to clean app keys such as ``gmail``."""
    slug = getattr(value, "slug", None)
    if slug:
        value = slug
    name = str(value).strip().lower()
    match = re.fullmatch(r"slug\s*=\s*['\"]([^'\"]+)['\"]", name)
    return match.group(1).strip().lower() if match else name


def _normalize_connected_apps(apps: dict[str, bool]) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    for app_name, is_connected in apps.items():
        clean_name = _normalize_app_name(app_name)
        if clean_name:
            normalized[clean_name] = normalized.get(clean_name, False) or is_connected
    return normalized


def _empty_profile(user_id: str) -> dict[str, Any]:
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


async def get_user_profile(user_id: str) -> dict | None:
    """Retrieve a profile and normalize its connected-app keys."""
    cached_profile = _get_cached_profile(user_id)
    if cached_profile is not None:
        return cached_profile

    db = get_db()
    if db is None:
        return None
    try:
        profile = await db.user_profiles.find_one({"_id": user_id})
        if profile and isinstance(profile.get("connected_apps"), dict):
            connected_apps = _normalize_connected_apps(profile["connected_apps"])
            if connected_apps != profile["connected_apps"]:
                await db.user_profiles.update_one({"_id": user_id}, {"$set": {"connected_apps": connected_apps}})
                profile["connected_apps"] = connected_apps
        if profile:
            _cache_profile(user_id, profile)
        return profile
    except Exception as error:
        logger.error("Error fetching profile for %s: %s", user_id, error)
        return None


async def get_or_create_user_profile(user_id: str) -> dict:
    """Get an existing profile or initialize a new one."""
    cached_profile = _get_cached_profile(user_id)
    if cached_profile is not None:
        return cached_profile

    db = get_db()
    if db is None:
        return _empty_profile(user_id)
    try:
        profile = await db.user_profiles.find_one({"_id": user_id})
        if profile:
            if isinstance(profile.get("connected_apps"), dict):
                profile["connected_apps"] = _normalize_connected_apps(profile["connected_apps"])
            _cache_profile(user_id, profile)
            return profile
        now = datetime.now(timezone.utc)
        new_profile = {**_empty_profile(user_id), "created_at": now, "updated_at": now}
        await db.user_profiles.insert_one(new_profile)
        _cache_profile(user_id, new_profile)
        return new_profile
    except Exception as error:
        logger.error("Error in get_or_create_user_profile for %s: %s", user_id, error)
        return _empty_profile(user_id)


async def update_user_profile(user_id: str, updates: dict[str, Any]) -> dict | None:
    """Update selected profile fields."""
    db = get_db()
    if db is None:
        return None
    updates["updated_at"] = datetime.now(timezone.utc)
    try:
        profile = await db.user_profiles.find_one_and_update(
            {"_id": user_id}, {"$set": updates}, upsert=True, return_document=True
        )
        if profile is not None:
            if isinstance(profile.get("connected_apps"), dict):
                profile["connected_apps"] = _normalize_connected_apps(profile["connected_apps"])
            _cache_profile(user_id, profile)
        else:
            _profile_cache.pop(user_id, None)
        return profile
    except Exception as error:
        logger.error("Error updating profile for %s: %s", user_id, error)
        return None
