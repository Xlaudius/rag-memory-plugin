#!/usr/bin/env python3
"""Simple migration script from ~/rag-system to RAG Memory plugin."""

import json
import sqlite3
import sys
from pathlib import Path

def migrate():
    """Migrate data from legacy ~/rag-system to plugin."""
    legacy_db = Path.home() / "rag-system" / "rag_data.db"
    new_db = Path.home() / ".hermes" / "plugins" / "rag-memory" / "rag_memory.db"

    if not legacy_db.exists():
        print(f"✗ Legacy database not found: {legacy_db}")
        sys.exit(1)

    print(f"📂 Legacy DB: {legacy_db}")
    print(f"📂 New DB: {new_db}")

    # Connect to legacy database
    conn = sqlite3.connect(str(legacy_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get documents from doc_vectors table
    try:
        cursor.execute("SELECT id, content, metadata, created_at FROM doc_vectors ORDER BY id")
        documents = cursor.fetchall()
        print(f"\n✓ Found {len(documents)} documents in legacy DB")
    except sqlite3.OperationalError:
        print("✗ Failed to read from doc_vectors table")
        conn.close()
        sys.exit(1)

    conn.close()

    # Import to new database
    try:
        from rag_memory.core import RAGCore

        new_db.parent.mkdir(parents=True, exist_ok=True)
        rag = RAGCore(str(new_db))

        imported = 0
        for doc in documents:
            try:
                metadata = json.loads(doc["metadata"]) if doc["metadata"] else {}
                metadata["legacy_id"] = doc["id"]
                metadata["migrated_at"] = str(Path.cwd())

                rag.add_document(
                    content=doc["content"],
                    namespace="legacy",
                    metadata=metadata,
                )
                imported += 1

                if imported % 10 == 0:
                    print(f"  Progress: {imported}/{len(documents)}")

            except Exception as e:
                print(f"⚠ Failed to import document {doc['id']}: {e}")

        print(f"\n✓ Imported {imported}/{len(documents)} documents")

        # Show stats
        stats = rag.get_stats()
        print(f"\n📊 New database stats:")
        print(f"  Documents: {stats.get('Documents', 0)}")
        print(f"  Namespaces: {stats.get('Namespaces', 0)}")

    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        sys.exit(1)

if __name__ == "__main__":
    migrate()
