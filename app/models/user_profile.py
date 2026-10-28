from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    user_id: str = Field(alias="_id")
    name: str | None = None
    email: str | None = None
    age: int | None = None
    location: str | None = None
    connected_apps: dict[str, bool] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)
    custom_attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True

