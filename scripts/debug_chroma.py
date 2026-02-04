#!/usr/bin/env python3
"""Debug Chroma retrieval: run a query and print raw results (distances, similarities, threshold).

Usage (from project root):
  python scripts/debug_chroma.py "I like espresso"
  CHROMA_PERSIST_DIR=./data/chroma python scripts/debug_chroma.py "fruity coffee"
"""
import json
import sys
import os

# Run from project root so app is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.vectorstore.chroma_store import ChromaStore


def main():
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "espresso"
    if not query:
        print("Usage: python scripts/debug_chroma.py \"your search query\"", file=sys.stderr)
        sys.exit(1)

    store = ChromaStore()
    store.initialize()

    debug = store.search_debug(query)
    print(json.dumps(debug, indent=2))


if __name__ == "__main__":
    main()
