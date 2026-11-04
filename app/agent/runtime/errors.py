"""Agent error types and provider error formatting."""

from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelConnectionError,
    ModelHTTPError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
)


class AgentBusyError(Exception):
    """Raised when the model provider is temporarily unable to serve a request."""


TRANSIENT_MODEL_ERRORS = (
    ModelRateLimitError,
    ModelUnavailableError,
    ModelServerError,
    ModelConnectionError,
    ModelTimeoutError,
)


def format_model_error_details(error: Exception) -> str:
    """Format provider and HTTP errors cleanly for terminal logging."""
    if isinstance(error, FallbackExceptionGroup):
        return "All fallback models failed: " + "; ".join(
            format_model_error_details(sub_error)
            for sub_error in error.exceptions
            if isinstance(sub_error, Exception)
        )
    if isinstance(error, ModelHTTPError):
        code = getattr(error, "status_code", "Unknown")
        body = getattr(error, "body", None)
        return f"HTTP {code} from provider | Response: {body}"
    if isinstance(error, ModelConnectionError):
        return f"Connection Failed: {error}"
    if isinstance(error, ModelTimeoutError):
        return f"Request Timeout: {error}"
    if isinstance(error, ModelRateLimitError):
        retry = getattr(error, "retry_after", None)
        return f"Rate Limit Exceeded (retry after {retry}s)" if retry else f"Rate Limit Exceeded: {error}"
    if isinstance(error, ModelQuotaExceededError):
        return f"Quota Exceeded: {error}"
    cause = getattr(error, "__cause__", None)
    if cause:
        return f"{type(error).__name__}: {error} [Root Cause: {type(cause).__name__}: {cause}]"
    return f"{type(error).__name__}: {error}"
