from pydantic_ai import RunContext

from ...utils.composio_config import AVAILABLE_APPS
from .context import Session


def inject_user_context(ctx: RunContext[Session]) -> str:
    """Add the current user's profile, integrations, and memories to the prompt."""
    prompts: list[str] = []
    available_integrations = ", ".join(
        f"{name} ({slug})" for slug, name in AVAILABLE_APPS.items()
    )
    prompts.append(f"Available integrations: {available_integrations}.")

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
    for key, value in custom_attrs.items():
        user_info.append(f"{key.capitalize()}: {value}")

    if user_info:
        prompts.append(f"User Profile: {', '.join(user_info)}")
    else:
        prompts.append(
            "User Profile: Unknown (if the user shares their name, age, location, "
            "or details, call update_profile to save them)."
        )

    if active_tools:
        prompts.append(
            f"Connected Integrations: {', '.join(active_tools)}. "
            "These tools are already authenticated; invoke them directly without "
            "prompting the user to connect again."
        )
    else:
        prompts.append(
            "Connected Integrations: None currently active. If the user requests "
            "an integration, use Composio's connection flow."
        )

    if ctx.deps.relevant_memories:
        memories = "\n".join(f"- {memory}" for memory in ctx.deps.relevant_memories)
        prompts.append(
            f"\n[Long-Term Memories about this user]:\n{memories}\n"
            "Use these memories naturally to personalize responses when relevant."
        )

    return "\n" + "\n".join(prompts) + "\n"
