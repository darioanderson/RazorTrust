from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from razortrust.database import SqlHoldRepository


async def export_feedback(output: Path, database_url: str) -> None:
    engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        repository = SqlHoldRepository(async_sessionmaker(engine, expire_on_commit=False))
        examples = await repository.export_analyst_training_examples()
    finally:
        await engine.dispose()
    content = "".join(
        json.dumps(example.model_dump(mode="json"), sort_keys=True) + "\n" for example in examples
    )
    if output.exists():
        raise FileExistsError("feedback exports are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "row_count": len(examples),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "analyst_identity_exported": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export adjudicated analyst labels for retraining")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database_url = os.getenv("RAZORTRUST_DATABASE_URL")
    if not database_url:
        raise ValueError("RAZORTRUST_DATABASE_URL is required")
    asyncio.run(export_feedback(args.output, database_url))


if __name__ == "__main__":
    main()
