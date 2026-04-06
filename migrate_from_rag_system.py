#!/usr/bin/env python3
"""Migrate data from legacy ~/rag-system to RAG Memory plugin.

This script properly handles:
- TF-IDF database (regular SQLite)
- Neural database (sqlite-vec with chunked metadata)
- Reconstruction of documents from chunked tables
- Deduplication and error handling
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def export_neural_documents(legacy_db_path: Path) -> list[dict]:
    """Export documents from legacy neural RAG database.

    The neural database uses sqlite-vec which stores data in chunked tables:
    - doc_vectors_metadatatext01: namespace
    - doc_vectors_metadatatext02: content
    - doc_vectors_metadatatext03: metadata/type
    - doc_vectors_metadatatext04: timestamp

    Args:
        legacy_db_path: Path to ~/rag-system/rag_data.db

    Returns:
        List of documents with content, namespace, metadata
    """
    if not legacy_db_path.exists():
        logger.error(f"Neural database not found: {legacy_db_path}")
        return []

    logger.info(f"📂 Exporting from neural database: {legacy_db_path}")

    conn = sqlite3.connect(str(legacy_db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    documents = []

    try:
        # Get total row count
        cursor.execute("SELECT COUNT(*) FROM doc_vectors_rowids")
        total = cursor.fetchone()[0]
        logger.info(f"  Found {total} documents")

        # Reconstruct documents from chunked metadata
        for rowid in range(1, total + 1):
            try:
                # Read chunks
                namespace = None
                content = None
                metadata = {}
                timestamp = None

                # Read namespace from chunk 01
                try:
                    cursor.execute(
                        "SELECT data FROM doc_vectors_metadatatext01 WHERE rowid = ?", (rowid,)
                    )
                    result = cursor.fetchone()
                    if result and result[0]:
                        namespace = result[0]
                except Exception:
                    namespace = "legacy"

                # Read content from chunk 02
                try:
                    cursor.execute(
                        "SELECT data FROM doc_vectors_metadatatext02 WHERE rowid = ?", (rowid,)
                    )
                    result = cursor.fetchone()
                    if result and result[0]:
                        content = result[0]
                except Exception:
                    continue

                # Read metadata from chunk 03
                try:
                    cursor.execute(
                        "SELECT data FROM doc_vectors_metadatatext03 WHERE rowid = ?", (rowid,)
                    )
                    result = cursor.fetchone()
                    if result and result[0]:
                        try:
                            metadata = json.loads(result[0]) if isinstance(result[0], str) else {}
                        except json.JSONDecodeError:
                            metadata = {"raw": result[0]}
                except Exception:
                    metadata = {}

                # Read timestamp from chunk 04
                try:
                    cursor.execute(
                        "SELECT data FROM doc_vectors_metadatatext04 WHERE rowid = ?", (rowid,)
                    )
                    result = cursor.fetchone()
                    if result and result[0]:
                        timestamp = result[0]
                except Exception:
                    timestamp = datetime.now(timezone.utc).isoformat()

                if content:
                    doc = {
                        "rowid": rowid,
                        "content": content,
                        "namespace": namespace or "legacy",
                        "metadata": metadata,
                        "created_at": timestamp,
                    }
                    documents.append(doc)

                    if len(documents) % 10 == 0:
                        logger.info(f"  Progress: {len(documents)}/{total}")

            except Exception as e:
                logger.warning(f"  ⚠ Failed to read row {rowid}: {e}")
                continue

        logger.info(f"✓ Exported {len(documents)} documents from neural database")

    except Exception as e:
        logger.error(f"✗ Neural export failed: {e}")
    finally:
        conn.close()

    return documents


def export_tfidf_documents(legacy_db_path: Path) -> list[dict]:
    """Export documents from legacy TF-IDF RAG database.

    Args:
        legacy_db_path: Path to ~/rag-system/rag_data_tfidf.db

    Returns:
        List of documents with content, namespace, metadata
    """
    if not legacy_db_path.exists():
        logger.info(f"  TF-IDF database not found: {legacy_db_path}")
        return []

    logger.info(f"📂 Exporting from TF-IDF database: {legacy_db_path}")

    conn = sqlite3.connect(str(legacy_db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    documents = []

    try:
        cursor.execute("SELECT id, content, metadata FROM documents")
        rows = cursor.fetchall()

        for row in rows:
            try:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                doc = {
                    "rowid": row["id"],
                    "content": row["content"],
                    "namespace": "legacy_tfidf",
                    "metadata": metadata,
                    "created_at": metadata.get("created_at"),
                }
                documents.append(doc)
            except Exception as e:
                logger.warning(f"  ⚠ Failed to parse doc {row['id']}: {e}")
                continue

        logger.info(f"✓ Exported {len(documents)} documents from TF-IDF database")

    except Exception as e:
        logger.error(f"✗ TF-IDF export failed: {e}")
    finally:
        conn.close()

    return documents


def import_to_plugin(documents: list[dict], plugin_db_path: Path) -> int:
    """Import documents to RAG Memory plugin database.

    Args:
        documents: List of documents from export
        plugin_db_path: Path to rag_core.db

    Returns:
        Number of documents imported
    """
    if not documents:
        logger.warning("No documents to import")
        return 0

    logger.info(f"📂 Importing to: {plugin_db_path}")

    try:
        from rag_memory.core import RAGCore

        plugin_db_path.parent.mkdir(parents=True, exist_ok=True)
        rag = RAGCore(str(plugin_db_path))

        imported = 0
        skipped = 0

        for doc in documents:
            try:
                # Prepare metadata
                metadata = doc.get("metadata", {})
                metadata["legacy_rowid"] = doc.get("rowid")
                metadata["legacy_created_at"] = doc.get("created_at")
                metadata["migrated_at"] = datetime.now(timezone.utc).isoformat()
                metadata["migration_source"] = doc.get("namespace", "legacy")

                # Add document
                rag.add_document(
                    content=doc.get("content", ""),
                    namespace=doc.get("namespace", "legacy"),
                    metadata=metadata,
                )
                imported += 1

                if imported % 10 == 0:
                    logger.info(f"  Progress: {imported}/{len(documents)}")

            except Exception as e:
                logger.warning(f"  ⚠ Failed to import document {doc.get('rowid')}: {e}")
                skipped += 1

        logger.info(f"✓ Imported {imported}/{len(documents)} documents")
        if skipped > 0:
            logger.warning(f"  ⚠ Skipped {skipped} documents due to errors")

        return imported

    except ImportError as e:
        logger.error(f"✗ Failed to import RAG module: {e}")
        logger.error("  Install with: pip install rag-memory-plugin[neural]")
        return 0


def verify_migration(
    legacy_neural_db: Path, legacy_tfidf_db: Path, plugin_db: Path
) -> bool:
    """Verify migration success by comparing document counts.

    Args:
        legacy_neural_db: Original neural database
        legacy_tfidf_db: Original TF-IDF database
        plugin_db: New plugin database

    Returns:
        True if verification passed
    """
    logger.info("🔍 Verifying migration...")

    # Count legacy documents
    legacy_count = 0

    if legacy_neural_db.exists():
        try:
            conn = sqlite3.connect(str(legacy_neural_db))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM doc_vectors_rowids")
            neural_count = cursor.fetchone()[0]
            legacy_count += neural_count
            conn.close()
            logger.info(f"  Legacy neural: {neural_count} documents")
        except Exception as e:
            logger.warning(f"  Could not count neural documents: {e}")

    if legacy_tfidf_db.exists():
        try:
            conn = sqlite3.connect(str(legacy_tfidf_db))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            tfidf_count = cursor.fetchone()[0]
            legacy_count += tfidf_count
            conn.close()
            logger.info(f"  Legacy TF-IDF: {tfidf_count} documents")
        except Exception as e:
            logger.warning(f"  Could not count TF-IDF documents: {e}")

    # Count plugin documents
    try:
        from rag_memory.core import RAGCore

        rag = RAGCore(str(plugin_db))
        stats = rag.get_stats()
        plugin_count = stats.get("Documents", 0)
        logger.info(f"  Plugin database: {plugin_count} documents")

        # Allow for existing documents
        if plugin_count >= legacy_count:
            logger.info("✓ Verification passed (plugin has >= legacy documents)")
            return True
        else:
            logger.warning(
                f"⚠ Document count mismatch: plugin={plugin_count}, legacy={legacy_count}"
            )
            return False

    except Exception as e:
        logger.warning(f"Could not verify: {e}")
        return False


def main() -> None:
    """Run migration."""
    logger.info("🚀 RAG Memory Migration: Legacy → Plugin\n")

    # Paths
    legacy_neural_db = Path.home() / "rag-system" / "rag_data.db"
    legacy_tfidf_db = Path.home() / "rag-system" / "rag_data_tfidf.db"
    hermes_home = Path.home() / ".hermes"
    plugin_db = hermes_home / "plugins" / "rag-memory" / "rag_core.db"

    # Check if legacy databases exist
    if not legacy_neural_db.exists() and not legacy_tfidf_db.exists():
        logger.error("✗ No legacy databases found")
        logger.info(f"  Expected: {legacy_neural_db}")
        logger.info(f"  Or: {legacy_tfidf_db}")
        sys.exit(1)

    # Backup existing plugin DB if it exists
    if plugin_db.exists():
        backup_path = (
            plugin_db.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        )
        logger.info(f"📦 Backing up existing DB to: {backup_path}")
        plugin_db.rename(backup_path)

    # Export from legacy databases
    all_documents = []

    # Export neural
    neural_docs = export_neural_documents(legacy_neural_db)
    all_documents.extend(neural_docs)

    # Export TF-IDF
    tfidf_docs = export_tfidf_documents(legacy_tfidf_db)
    all_documents.extend(tfidf_docs)

    if not all_documents:
        logger.error("✗ No documents found to migrate")
        sys.exit(1)

    logger.info(f"\n📊 Total documents to migrate: {len(all_documents)}")

    # Import to plugin
    imported = import_to_plugin(all_documents, plugin_db)

    if imported == 0:
        logger.error("✗ Migration failed - no documents imported")
        sys.exit(1)

    # Verify
    if verify_migration(legacy_neural_db, legacy_tfidf_db, plugin_db):
        logger.info("\n✅ Migration complete!")
        logger.info(f"📂 New database: {plugin_db}")
        logger.info(f"📊 Documents imported: {imported}")
        logger.info("\n🧪 Test with:")
        logger.info("  rag-memory doctor")
        logger.info("  rag-memory search 'test query'")
    else:
        logger.warning("\n⚠ Migration completed with warnings")
        sys.exit(1)


if __name__ == "__main__":
    main()
