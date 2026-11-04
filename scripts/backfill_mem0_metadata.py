# Tag existing Mem0 memories with the episodic metadata marker.
import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.mem0_config import backfill_episodic_metadata


async def run(user_ids: list[str], dry_run: bool) -> None:
    for user_id in user_ids:
        updated = await backfill_episodic_metadata(user_id, dry_run=dry_run)
        action = "would tag" if dry_run else "tagged"
        print(f"{user_id}: {action} {updated} memories")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Add metadata={'kind': 'episodic'} to existing Mem0 memories."
    )
    parser.add_argument("user_ids", nargs="+", help="User IDs to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Report without updating")
    args = parser.parse_args()
    asyncio.run(run(args.user_ids, args.dry_run))


if __name__ == "__main__":
    main()
