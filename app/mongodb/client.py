import logging
import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

load_dotenv()

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(86400 * 2)))

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None
_ttl_initialized = False


def get_db() -> AsyncIOMotorDatabase | None:
    """Lazily initialize and return the async MongoDB database instance."""
    global _client, _db
    if _db is not None:
        return _db

    if not MONGODB_URI:
        logger.warning("MONGODB_URI not found in environment variables.")
        return None

    try:
        _client = AsyncIOMotorClient(MONGODB_URI)
        _db = _client[DB_NAME]
        return _db
    except Exception as error:
        logger.error("Failed to connect to MongoDB: %s", error)
        return None


async def ensure_ttl_index() -> None:
    """Ensure old chat sessions expire automatically."""
    global _ttl_initialized
    if _ttl_initialized:
        return
    db = get_db()
    if db is None:
        return
    try:
        await db.chat_sessions.create_index("updated_at", expireAfterSeconds=SESSION_TTL_SECONDS)
        _ttl_initialized = True
    except Exception as error:
        logger.warning("Could not initialize chat_sessions TTL index: %s", error)
