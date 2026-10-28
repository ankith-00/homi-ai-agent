import os
import asyncio
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..utils.composio_config import composio_mcp
from ..utils.mem0_config import add_memory, search_memories
from ..utils.mongo_config import (
    get_or_create_user_profile,
    update_user_profile,
    sync_composio_connections,
    load_session_history,
    save_session_history,
    delete_session_history,
)

load_dotenv()

MAX_MESSAGES = 30


@dataclass
class Session:
    user_id: str = ""
    profile: dict = field(default_factory=dict)
    history: list[ModelMessage] = field(default_factory=list)
    relevant_memories: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


sessions: dict[str, Session] = {}


# Model Configuration
model = OpenAIChatModel(
    os.getenv("NVIDIA_MODEL"),
    provider=OpenAIProvider(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    ),
)

INSTRUCTIONS = (Path(__file__).parent / "instructions.md").read_text(encoding="utf-8")

# Agent Configuration
agent = Agent(
    model,
    deps_type=Session,
    system_prompt=INSTRUCTIONS,
)



@agent.system_prompt
def inject_user_context(ctx: RunContext[Session]) -> str:
    """
    Inject user profile (name, age, location, attributes, auth status) and relevant long-term memories.
    """
    prompts: list[str] = []
    profile = ctx.deps.profile
    name = profile.get("name")
    age = profile.get("age")
    location = profile.get("location")
    custom_attrs = profile.get("custom_attributes", {})
    connected_apps = profile.get("connected_apps", {})
    active_tools = [app for app, active in connected_apps.items() if active]

    user_info = []
    if name:
        user_info.append(f"Name: {name}")
    if age:
        user_info.append(f"Age: {age}")
    if location:
        user_info.append(f"Location: {location}")
    for k, v in custom_attrs.items():
        user_info.append(f"{k.capitalize()}: {v}")

    if user_info:
        prompts.append(f"User Profile: {', '.join(user_info)}")
    else:
        prompts.append("User Profile: Unknown (if user shares their name, age, location, or details, call update_profile to save them).")

    if active_tools:
        prompts.append(
            f"Connected Integrations: {', '.join(active_tools)}. "
            "These tools are already authenticated; invoke them directly without prompting the user to connect again."
        )
    else:
        prompts.append("Connected Integrations: None currently active. If user requests email actions, prompt them to connect.")

    if ctx.deps.relevant_memories:
        memories_str = "\n".join(f"- {m}" for m in ctx.deps.relevant_memories)
        prompts.append(
            f"\n[Long-Term Memories about this user]:\n"
            f"{memories_str}\n"
            f"Use these memories naturally to personalize your responses when relevant."
        )

    return "\n" + "\n".join(prompts) + "\n"




@agent.tool
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
    """
    Update the user's profile in MongoDB when they provide details about themselves.

    Args:
        name: User's name (e.g. "Ankith")
        age: User's age as an integer (e.g. 25)
        location: User's location/city/country (e.g. "Bengaluru, India")
        preference_key: Specific preference category (e.g. "email_tone")
        preference_value: Value for that preference (e.g. "concise")
        custom_field: Any other attribute name (e.g. "occupation", "company", "hobbies")
        custom_value: Value for that attribute (e.g. "Software Engineer")
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
        prefs = profile.setdefault("preferences", {})
        prefs[preference_key] = preference_value
        updates["preferences"] = prefs
    if custom_field and custom_value:
        attrs = profile.setdefault("custom_attributes", {})
        cleaned_key = custom_field.strip().lower()
        attrs[cleaned_key] = custom_value
        updates[f"custom_attributes.{cleaned_key}"] = custom_value

    if updates:
        await update_user_profile(ctx.deps.user_id, updates)
        return f"Updated user profile with: {', '.join(updates.keys())}"
    return "No profile fields to update."




@agent.tool
async def refresh_connected_apps(ctx: RunContext[Session]) -> str:
    """
    Refresh and sync the user's active connected integrations from Composio.
    """
    connected = await sync_composio_connections(ctx.deps.user_id)
    ctx.deps.profile["connected_apps"] = connected
    active = [app for app, is_act in connected.items() if is_act]
    if active:
        return f"Connected apps updated: {', '.join(active)}"
    return "No active integrations found."




def trim(history: list[ModelMessage]) -> list[ModelMessage]:
    history = history[-MAX_MESSAGES:]
    while history and not (
        isinstance(history[0], ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in history[0].parts)
    ):
        history = history[1:]
    return history




async def run_agent(session_id: str, user_id: str, text: str) -> str:
    # 1. Load persisted chat history 
    history = await load_session_history(user_id, session_id)

    # 2. Fetch fresh user profile
    profile = await get_or_create_user_profile(user_id)
    if not profile.get("connected_apps"):
        profile["connected_apps"] = await sync_composio_connections(user_id)

    # 3. Retrieve relevant long-term memories for this specific user from Mem0
    relevant_memories = await search_memories(query=text, user_id=user_id, top_k=5)

    # 4. Construct session deps for the agent
    session = Session(
        user_id=user_id,
        profile=profile,
        history=history,
        relevant_memories=relevant_memories,
    )

    # 5. Run the agent with this user's context, profile & tools
    result = await agent.run(
        text,
        deps=session,
        message_history=history,
        toolsets=[composio_mcp(user_id)],
    )

    # 6. Trim & persist updated chat history to MongoDB with auto-expiring TTL
    updated_history = trim(result.all_messages())
    await save_session_history(user_id, session_id, updated_history)

    # 7. Record the turn in Mem0 tied strictly to this user_id (non-blocking)
    asyncio.create_task(
        add_memory(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": result.output},
            ],
            user_id=user_id,
        )
    )

    return result.output