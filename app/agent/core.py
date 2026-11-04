"""Public interface for the Homi agent runtime.

The implementation lives in ``app.agent.runtime``; this module keeps the
historical import surface stable for routers and external callers.
"""

from .runtime import (
    INSTRUCTIONS,
    AgentBusyError,
    AgentStreamChunk,
    agent,
    build_agent_system_prompt,
    model,
    prepare_session,
    run_agent,
    sessions,
    stream_agent,
)

__all__ = [
    "INSTRUCTIONS",
    "AgentBusyError",
    "AgentStreamChunk",
    "agent",
    "build_agent_system_prompt",
    "model",
    "prepare_session",
    "run_agent",
    "sessions",
    "stream_agent",
]
