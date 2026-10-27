import os
import asyncio
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

# Agent Configuration
agent = Agent(
    model,
    deps_type=Session,
    system_prompt="You are a helpful voice assistant. Keep replies short and conversational. Always confirm with the user before sending or deleting any email.",
)


@agent.system_prompt
def inject_user_context(ctx: RunContext[Session]) -> str:
    """Inject user profile (name, auth status) and relevant long-term memories."""
    prompts: list[str] = []
    profile = ctx.deps.profile
    name = profile.get("name")
    connected_apps = profile.get("connected_apps", {})
    active_tools = [app for app, active in connected_apps.items() if active]

    if name:
        prompts.append(f"User's Name: {name}")
    else:
        prompts.append("User's Name: Unknown (if the user shares their name, call the update_profile tool to save it).")

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
    preference_key: str | None = None,
    preference_value: str | None = None,
) -> str:
    """
    Update the user's profile in MongoDB when they provide their name or preferences.
    """
    updates = {}
    if name:
        ctx.deps.profile["name"] = name
        updates["name"] = name
    if preference_key and preference_value:
        prefs = ctx.deps.profile.setdefault("preferences", {})
        prefs[preference_key] = preference_value
        updates["preferences"] = prefs

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
    # 1. Load persisted chat history from MongoDB (persists across server restarts & workers)
    history = await load_session_history(user_id, session_id)

    # 2. Always fetch fresh user profile from MongoDB for this exact user_id
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

