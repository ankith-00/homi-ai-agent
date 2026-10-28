import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..utils.composio_config import composio_mcp
from ..utils.mem0_config import add_memory, search_memories
from ..utils.mongo_config import (
    get_or_create_user_profile,
    sync_composio_connections,
    load_session_history,
    save_session_history,
)
from .components.context import Session
from .components.prompts import inject_user_context
from .components.tools import AGENT_TOOLS

load_dotenv()

MAX_MESSAGES = 30


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

for tool_function in AGENT_TOOLS:
    agent.tool(tool_function)
agent.system_prompt(inject_user_context)
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
