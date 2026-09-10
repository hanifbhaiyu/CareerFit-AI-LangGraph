"""Build the FAISS index from the knowledge base.

Run once before starting the API:

    python -m backend.rag.ingest --force
"""

from __future__ import annotations

import argparse
import logging

from backend.config import get_settings
from backend.core.logging_config import setup_logging
from backend.rag.vector_store import build_vector_store, load_knowledge_base_documents, split_documents

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CareerFit knowledge index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if an index already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show chunking statistics without calling the embedding API.",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    if args.dry_run:
        chunks = split_documents(load_knowledge_base_documents())
        sizes = [len(chunk.page_content) for chunk in chunks]
        print(f"Documents : {len({c.metadata['source'] for c in chunks})}")
        print(f"Chunks    : {len(chunks)}")
        print(f"Chunk size: min={min(sizes)} avg={sum(sizes) // len(sizes)} max={max(sizes)}")
        print("\nFirst chunk preview:\n" + chunks[0].page_content[:300])
        return

    problems = settings.validate()
    if problems:
        for problem in problems:
            logger.error(problem)
        raise SystemExit(1)

    build_vector_store(force=args.force)
    print(f"Index written to {settings.vector_store_dir}")


if __name__ == "__main__":
    main()
