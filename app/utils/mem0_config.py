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

EPISODIC_MEMORY_METADATA = {"kind": "episodic"}
EPISODIC_EXTRACTION_INSTRUCTIONS = (
    "Store only durable episodic context: ongoing projects, events, relationships, "
    "decisions, and conversational context. Do not store identity or profile facts "
    "(name, age, location, timezone, connected apps, explicit preferences); those are "
    "managed by the user profile store and are authoritative there."
)

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
        response = await client.add(
            messages,
            user_id=user_id,
            metadata=dict(EPISODIC_MEMORY_METADATA),
            custom_instructions=EPISODIC_EXTRACTION_INSTRUCTIONS,
        )
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
            metadata=dict(EPISODIC_MEMORY_METADATA),
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


async def backfill_episodic_metadata(
    user_id: str,
    *,
    page_size: int = 100,
    dry_run: bool = False,
) -> int:
    """Tag legacy Mem0 memories with the episodic metadata marker.

    Returns the number of memories that were (or would be) updated.
    """
    client = get_mem0_client()
    if not client:
        return 0

    updated = 0
    page = 1
    seen_ids: set[str] = set()

    while True:
        try:
            response = await client.get_all(
                filters={"user_id": user_id},
                page=page,
                page_size=page_size,
            )
        except Exception as e:
            logger.error(f"Failed to list Mem0 memories for user {user_id}: {e}")
            break

        results = response.get("results", []) if isinstance(response, dict) else []
        if not results:
            break

        page_ids = {
            str(memory_id)
            for item in results
            if (memory_id := item.get("id") or item.get("memory_id"))
        }
        if page > 1 and (not page_ids or page_ids.issubset(seen_ids)):
            break
        seen_ids.update(page_ids)

        for item in results:
            metadata = item.get("metadata")
            if isinstance(metadata, dict) and metadata.get("kind"):
                continue
            memory_id = item.get("id") or item.get("memory_id")
            if not memory_id:
                continue
            merged_metadata = {
                **(metadata if isinstance(metadata, dict) else {}),
                **EPISODIC_MEMORY_METADATA,
            }
            if dry_run:
                updated += 1
                continue
            try:
                await client.update(str(memory_id), metadata=merged_metadata)
                updated += 1
            except Exception as e:
                logger.error(f"Failed to tag Mem0 memory {memory_id} for user {user_id}: {e}")

        if len(results) < page_size:
            break
        page += 1

    logger.info(
        "Mem0 episodic backfill user=%s updated=%d dry_run=%s",
        user_id,
        updated,
        dry_run,
    )
    return updated

