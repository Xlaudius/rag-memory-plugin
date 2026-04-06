#!/usr/bin/env python3
"""Deployment verification script for RAG Memory plugin.

Tests that:
1. Config is correctly set
2. Plugin loads successfully
3. Database is accessible
4. Basic operations work
5. File indexing works
"""

import os
import sys
import yaml
from pathlib import Path

print("="*60)
print("RAG Memory Plugin - Deployment Verification")
print("="*60 + "\n")

# Test 1: Config verification
print("Test 1: Config verification...")
config_path = Path.home() / ".hermes" / "config.yaml"

if not config_path.exists():
    print("  ✗ Config file not found")
    sys.exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

if "plugins" not in config:
    print("  ✗ No plugins section in config")
    sys.exit(1)

if "rag_memory" not in config["plugins"]:
    print("  ✗ No rag_memory plugin config")
    sys.exit(1)

rag_config = config["plugins"]["rag_memory"]

# Verify settings
assert rag_config["enabled"] == True, "Plugin should be enabled"
assert rag_config["mode"] == "hybrid", "Mode should be hybrid"
assert rag_config["auto_index_files"] == True, "Auto index should be enabled"
assert rag_config["index_on_session_start"] == True, "Session start index should be enabled"
assert rag_config["file_chunk_size"] == 2000, "Chunk size should be 2000"

print("  ✓ Config is correctly set")
print(f"    - enabled: {rag_config['enabled']}")
print(f"    - mode: {rag_config['mode']}")
print(f"    - auto_index_files: {rag_config['auto_index_files']}")
print(f"    - index_on_session_start: {rag_config['index_on_session_start']}")
print(f"    - file_chunk_size: {rag_config['file_chunk_size']}")

# Test 2: Plugin import
print("\nTest 2: Plugin import...")
try:
    from rag_memory.core import RAGCore
    from rag_memory.core.file_indexing import FileIndexer, chunk_by_headers, compute_hash
    print("  ✓ Plugin imports successfully")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Database access
print("\nTest 3: Database access...")
hermes_home = Path.home() / ".hermes"
data_dir = hermes_home / "plugins" / "rag-memory"
db_path = data_dir / "rag_core.db"

if not db_path.exists():
    print(f"  ✗ Database not found at {db_path}")
    sys.exit(1)

try:
    rag = RAGCore(str(db_path))
    print(f"  ✓ Database accessible: {db_path}")
except Exception as e:
    print(f"  ✗ Database access failed: {e}")
    sys.exit(1)

# Test 4: Basic operations
print("\nTest 4: Basic operations...")

# Add document
result = rag.add_document(
    content="Deployment test document",
    namespace="deployment_test",
    metadata={"test": True}
)
assert result is not None, "Failed to add document"
doc_id = result.get("id") if isinstance(result, dict) else result
print(f"  ✓ Add document: {doc_id}")

# Search
results = rag.search("deployment test", namespace="deployment_test")
assert len(results) > 0, "Search returned no results"
print(f"  ✓ Search: found {len(results)} results")

# Get document
doc = rag.get_document(doc_id)
assert doc is not None, "Failed to get document"
assert doc["content"] == "Deployment test document", "Content mismatch"
print(f"  ✓ Get document: {doc['content'][:30]}...")

# List namespaces
namespaces = rag.list_namespaces()
assert "deployment_test" in namespaces, "Namespace not found"
print(f"  ✓ List namespaces: found {len(namespaces)} namespaces")

# Get stats
stats = rag.get_stats()
assert stats["documents"] > 0, "No documents in stats"
print(f"  ✓ Get stats: {stats['documents']} documents, {stats['namespaces']} namespaces")

# Clean up test data
rag.delete_document(doc_id)
print("  ✓ Delete test document")

# Test 5: File indexing
print("\nTest 5: File indexing...")

# Create test files
test_dir = hermes_home / "test_indexing"
test_dir.mkdir(exist_ok=True)

test_file = test_dir / "test.md"
test_file.write_text("# Test\n\nContent for indexing")

indexer = FileIndexer(rag, hermes_home)

# Chunking test
chunks = chunk_by_headers("# Main\n\n## Section 1\n\nContent 1\n\n## Section 2\n\nContent 2")
assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
print("  ✓ Chunking: works correctly")

# Hash test
hash1 = compute_hash("test content")
hash2 = compute_hash("test content")
assert hash1 == hash2, "Hash not consistent"
print("  ✓ Hashing: works correctly")

# File indexing test
added = indexer.index_file(test_file, "test_files")
assert added > 0, "Failed to index file"
print(f"  ✓ File indexing: indexed {added} chunks")

# Search indexed content (without namespace filter)
results = rag.search("Content for indexing")
assert len(results) > 0, "File content not found in search"
print(f"  ✓ Search indexed files: found {len(results)} results")

# Clean up
test_file.unlink()
test_dir.rmdir()

# Test 6: Edge cases
print("\nTest 6: Edge cases...")

# Empty content
doc_id = rag.add_document("", namespace="test")
assert doc_id is not None, "Failed to add empty document"
print("  ✓ Empty content: handled")

# Special characters
special_content = "Special: ' \" ` \t\n\nEmoji: 🌍 ✓"
doc_id = rag.add_document(special_content, namespace="test")
assert doc_id is not None, "Failed to add special characters"
print("  ✓ Special characters: handled")

# Unicode
unicode_content = "Unicode: 世界 العربية Привет"
doc_id = rag.add_document(unicode_content, namespace="test")
assert doc_id is not None, "Failed to add unicode"
print("  ✓ Unicode: handled")

# Search empty
results = rag.search("", namespace="test")
assert isinstance(results, list), "Empty search should return list"
print("  ✓ Empty search: handled")

# Summary
print("\n" + "="*60)
print("Deployment Verification: ✓ ALL TESTS PASSED")
print("="*60)

print("\n🎉 RAG Memory Plugin is successfully deployed!")
print("\nConfiguration:")
print(f"  - Config: {config_path}")
print(f"  - Database: {db_path}")
print(f"  - Mode: {rag_config['mode']}")
print(f"  - Auto-index: {rag_config['auto_index_files']}")
print(f"  - Session start index: {rag_config['index_on_session_start']}")

print("\nNext steps:")
print("  1. Restart Hermes Agent to load new config")
print("  2. File indexing will run automatically on session start")
print("  3. Cron job will run every 4 hours")

sys.exit(0)
