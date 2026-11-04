"""Model, instruction, and agent configuration."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ...observability import configure_observability, content_for_trace
from ..components.context import Session
from ..components.prompts import inject_user_context
from ..components.tools import AGENT_TOOLS
from .errors import TRANSIENT_MODEL_ERRORS

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _build_model() -> tuple[Model, str]:
    """Build the primary model and, when configured, its fallback chain."""
    primary_model_name = os.getenv("NVIDIA_MODEL", "")
    primary_base_url = os.getenv("NVIDIA_BASE_URL", "").strip() or DEFAULT_BASE_URL
    primary_model = OpenAIChatModel(
        primary_model_name,
        provider=OpenAIProvider(
            base_url=primary_base_url,
            api_key=os.environ["NVIDIA_API_KEY"],
        ),
    )

    fallback_model_name = os.getenv("FALLBACK_MODEL", "").strip()
    if not fallback_model_name:
        return primary_model, primary_model_name

    fallback_base_url = os.getenv("FALLBACK_BASE_URL", "").strip() or primary_base_url
    fallback_api_key = os.getenv("FALLBACK_API_KEY", "").strip() or os.environ["NVIDIA_API_KEY"]
    fallback_model = OpenAIChatModel(
        fallback_model_name,
        provider=OpenAIProvider(base_url=fallback_base_url, api_key=fallback_api_key),
    )
    if fallback_model_name == primary_model_name and fallback_base_url == primary_base_url:
        logger.warning(
            "Fallback model is identical to the primary model and base URL; rate limits will likely still apply"
        )
    return (
        FallbackModel(primary_model, fallback_model, fallback_on=TRANSIENT_MODEL_ERRORS),
        f"{primary_model_name} -> {fallback_model_name}",
    )


model, MODEL_LABEL = _build_model()

INSTRUCTIONS = (Path(__file__).parent.parent / "instructions.md").read_text(encoding="utf-8")
logger.info(
    "Agent instructions loaded chars=%d content=%s",
    len(INSTRUCTIONS),
    json.dumps(content_for_trace(INSTRUCTIONS), ensure_ascii=False),
)


def build_agent_system_prompt(ctx: RunContext[Session]) -> str:
    """Render the static instruction template and per-user context per turn."""
    profile = ctx.deps.profile
    custom_attrs = profile.get("custom_attributes", {})
    preferences = profile.get("preferences", {})
    timezone_name = (
        profile.get("timezone")
        or (preferences.get("timezone") if isinstance(preferences, dict) else None)
        or (custom_attrs.get("timezone") if isinstance(custom_attrs, dict) else None)
        or "UTC"
    )
    if not isinstance(timezone_name, str):
        timezone_name = "UTC"
    try:
        user_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            "Unknown profile timezone turn=%s timezone=%s; using UTC",
            ctx.deps.turn_id,
            timezone_name,
        )
        timezone_name = "UTC"
        user_timezone = ZoneInfo("UTC")

    user_name = " ".join(str(profile.get("name") or "the user").split())[:80]
    now = datetime.now(user_timezone)
    rendered_instructions = (
        INSTRUCTIONS.replace("{{user_name}}", str(user_name))
        .replace("{{current_datetime}}", now.strftime("%Y-%m-%d %H:%M:%S %Z"))
        .replace("{{timezone}}", timezone_name)
    )
    prompt = rendered_instructions + inject_user_context(ctx)
    logger.info(
        "System prompt composed turn=%s chars=%d timezone=%s content=%s",
        ctx.deps.turn_id,
        len(prompt),
        timezone_name,
        json.dumps(content_for_trace(prompt), ensure_ascii=False),
    )
    return prompt


agent = Agent(model, deps_type=Session)

for tool_function in AGENT_TOOLS:
    agent.tool(tool_function)
agent.system_prompt(dynamic=True)(build_agent_system_prompt)
configure_observability(agent)
logger.info(
    "Agent startup ready instructions=loaded instruction_chars=%d model=%s registered_tools=%s composio_mcp=configured_per_user_on_first_request",
    len(INSTRUCTIONS),
    MODEL_LABEL,
    [tool_function.__name__ for tool_function in AGENT_TOOLS],
)
