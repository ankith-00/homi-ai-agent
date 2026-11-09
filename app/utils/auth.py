import os
import secrets
import logging

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

BEARER_PREFIX = "bearer "
API_KEY_HEADER = "X-API-Key"


async def api_key_auth(request: Request) -> None:
    """
    Verify the request carries the shared API key when one is configured.

    Accepts the key in the `X-API-Key` header or as an `Authorization: Bearer <key>`
    token (the format the web-tools Next.js proxy already sends). When
    AI_AGENT_API_KEY is not set, endpoints stay open (local dev) with a warning.
    """
    configured = os.getenv("AI_AGENT_API_KEY", "").strip()
    if not configured:
        logger.warning(
            "AI_AGENT_API_KEY is not set; endpoint %s is open. Set it to require authentication.",
            request.url.path,
        )
        return

    provided = request.headers.get(API_KEY_HEADER, "").strip()
    if not provided:
        authorization = request.headers.get("Authorization", "").strip()
        if authorization.lower().startswith(BEARER_PREFIX):
            provided = authorization[len(BEARER_PREFIX):].strip()

    if not provided or not secrets.compare_digest(provided, configured):
        logger.warning("Rejected unauthenticated request for %s.", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized access.",
        )
