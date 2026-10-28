from typing import Any

from pydantic_ai import RunContext

from ....utils.mongo_config import update_user_profile
from ..context import Session


async def update_profile(
    ctx: RunContext[Session],
    name: str | None = None,
    age: int | None = None,
    location: str | None = None,
    preference_key: str | None = None,
    preference_value: str | None = None,
    custom_field: str | None = None,
    custom_value: str | None = None,
) -> str:
    """Update the user's profile in MongoDB when they provide personal details.

    Args:
        name: User's name (e.g. "Ankith")
        age: User's age as an integer (e.g. 25)
        location: User's location/city/country (e.g. "Bengaluru, India")
        preference_key: A preference category (e.g. "email_tone")
        preference_value: The preference value (e.g. "concise")
        custom_field: Another attribute name (e.g. "occupation", "company", "hobbies")
        custom_value: The value for that attribute
    """
    updates: dict[str, Any] = {}
    profile = ctx.deps.profile

    if name:
        profile["name"] = name
        updates["name"] = name
    if age is not None:
        profile["age"] = age
        updates["age"] = age
    if location:
        profile["location"] = location
        updates["location"] = location
    if preference_key and preference_value:
        preferences = profile.setdefault("preferences", {})
        preferences[preference_key] = preference_value
        updates["preferences"] = preferences
    if custom_field and custom_value:
        attributes = profile.setdefault("custom_attributes", {})
        cleaned_key = custom_field.strip().lower()
        attributes[cleaned_key] = custom_value
        updates[f"custom_attributes.{cleaned_key}"] = custom_value

    if updates:
        await update_user_profile(ctx.deps.user_id, updates)
        return f"Updated user profile with: {', '.join(updates.keys())}"
    return "No profile fields to update."
