import os
from dotenv import load_dotenv

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..utils.composio_config import composio_mcp

load_dotenv()

MAX_MESSAGES = 30


@dataclass
class Session:
    history: list[ModelMessage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


sessions: dict[str, Session] = {}

# Model Configuration
model = OpenAIChatModel(
    os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
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

# @agent.tool_plain
# def get_current_time() -> str:
#     """Get the current UTC date and time."""
#     return datetime.now(timezone.utc).isoformat()


# @agent.tool
# def save_note(ctx: RunContext[Session], note: str) -> str:
#     """Save a note for the user in this session."""
#     ctx.deps.notes.append(note)
#     return "saved"


# @agent.tool
# def list_notes(ctx: RunContext[Session]) -> list[str]:
#     """List all notes saved in this session."""
#     return ctx.deps.notes


def trim(history: list[ModelMessage]) -> list[ModelMessage]:
    history = history[-MAX_MESSAGES:]
    while history and not (
        isinstance(history[0], ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in history[0].parts)
    ):
        history = history[1:]
    return history


async def run_agent(session_id: str, user_id: str, text: str) -> str:
    session = sessions.setdefault(session_id, Session())
    result = await agent.run(
        text,
        deps=session,
        message_history=session.history,
        toolsets=[composio_mcp(user_id)],
    )
    session.history = trim(result.all_messages())
    return result.output