"""Opt-in Logfire/OpenTelemetry tracing with safe-by-default content handling."""

import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_logfire: Any = None
CAPTURE_CHAT_CONTENT = os.getenv("LOGFIRE_CAPTURE_CHAT_CONTENT", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
MAX_CONTENT_ATTRIBUTE_CHARS = 4_000

_SECRET_VALUE = re.compile(
    r"(?i)([\"']?\b(?:access_token|refresh_token|api[_-]?key|client_secret|authorization)\b[\"']?\s*[:=]\s*[\"']?)[^\s,;&}\]\"']+"
)
_BEARER_VALUE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_AUTH_URL_MARKERS = (
    "oauth", "authorize", "composio", "connect.", "connection_id=", "code=", "state=", "token=",
)


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass


def configure_observability(agent: Any) -> None:
    """Enable Pydantic AI tracing when LOGFIRE_TOKEN is configured."""
    global _logfire
    token = os.getenv("LOGFIRE_TOKEN", "").strip()
    if not token:
        logger.info("Agent tracing is disabled; set LOGFIRE_TOKEN to enable Logfire.")
        return

    try:
        import logfire

        logfire.configure(
            token=token,
            send_to_logfire=True,
            console=False,
            service_name="homi-ai-agent",
            environment=os.getenv("ENVIRONMENT") or os.getenv("APP_ENV"),
        )
        # Keep model/tool spans useful for timing and flow inspection without
        # exporting prompts, completions, tool arguments, or tool return bodies.
        logfire.instrument_pydantic_ai(
            agent,
            include_content=False,
            include_binary_content=False,
        )
        _logfire = logfire
        logger.info("Pydantic AI tracing is enabled.")
    except Exception:
        logger.exception("Could not configure Logfire; agent tracing is disabled.")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a Logfire span, or a no-op span when tracing is disabled."""
    if _logfire is None:
        yield _NoopSpan()
        return
    with _logfire.span(name, **attributes) as active_span:
        yield active_span


def content_for_trace(value: str) -> str:
    """Redact common credentials/auth URLs and cap content stored as an attribute."""
    text = _BEARER_VALUE.sub("Bearer [REDACTED]", value)
    text = _SECRET_VALUE.sub(r"\1[REDACTED]", text)

    def redact_auth_url(match: re.Match[str]) -> str:
        url = match.group(0)
        return "[REDACTED_AUTH_URL]" if any(marker in url.lower() for marker in _AUTH_URL_MARKERS) else url

    text = _URL.sub(redact_auth_url, text)
    if len(text) > MAX_CONTENT_ATTRIBUTE_CHARS:
        text = text[:MAX_CONTENT_ATTRIBUTE_CHARS] + "…[truncated]"
    return text
