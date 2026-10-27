import os
import tempfile
import logging
from dotenv import load_dotenv

load_dotenv()

# Disable telemetry for production & privacy
os.environ.setdefault("MEM0_TELEMETRY", "false")

# Prevent Windows permission issues with default ~/.mem0 directory
if not os.environ.get("MEM0_DIR"):
    os.environ["MEM0_DIR"] = os.path.join(tempfile.gettempdir(), ".mem0")

from mem0 import AsyncMemoryClient

logger = logging.getLogger(__name__)

MEM0_API_KEY = os.getenv("MEM0_KEY") or os.getenv("MEM0_API_KEY")

_client: AsyncMemoryClient | None = None


def get_mem0_client() -> AsyncMemoryClient | None:
    """
    Lazily initializes and returns the AsyncMemoryClient.
    Prevents import-time crashes if network or key is missing.
    """
    global _client
    if _client is not None:
        return _client

    if not MEM0_API_KEY:
        logger.warning("MEM0_KEY or MEM0_API_KEY not found in environment.")
        return None

    try:
        _client = AsyncMemoryClient(api_key=MEM0_API_KEY)
        return _client
    except Exception as e:
        logger.error(f"Failed to initialize Mem0 client: {e}")
        return None


async def add_memory(messages: list[dict] | str, user_id: str) -> dict | None:
    """
    Save new user-assistant interaction or facts into Mem0.
    
    `messages` can be:
      - a single string: "User prefers concise replies."
      - or chat messages list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    client = get_mem0_client()
    if not client:
        return None
    try:
        response = await client.add(messages, user_id=user_id)
        return response
    except Exception as e:
        logger.error(f"Failed to add memory to Mem0 for user {user_id}: {e}")
        return None


async def search_memories(query: str, user_id: str, top_k: int = 5) -> list[str]:
    """
    Search relevant long-term memories for a given query and user.
    Returns a list of extracted memory strings.
    """
    client = get_mem0_client()
    if not client:
        return []
    try:
        response = await client.search(
            query=query,
            filters={"user_id": user_id},
            top_k=top_k,
        )
        results = response.get("results", []) if isinstance(response, dict) else []
        memories: list[str] = []
        for item in results:
            text = item.get("memory") or item.get("text")
            if text:
                memories.append(text)
        return memories
    except Exception as e:
        logger.error(f"Failed to search memories in Mem0 for user {user_id}: {e}")
        return []


async def get_all_memories(user_id: str) -> list[str]:
    """
    Retrieve all memories stored for a specific user.
    """
    client = get_mem0_client()
    if not client:
        return []
    try:
        response = await client.get_all(filters={"user_id": user_id})
        results = response.get("results", []) if isinstance(response, dict) else []
        memories: list[str] = []
        for item in results:
            text = item.get("memory") or item.get("text")
            if text:
                memories.append(text)
        return memories
    except Exception as e:
        logger.error(f"Failed to get memories from Mem0 for user {user_id}: {e}")
        return []

