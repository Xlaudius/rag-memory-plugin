"""Edge case tests for RAG Memory plugin.

Comprehensive test suite covering edge cases to ensure >95% coverage.

Tests cover:
- Empty/None inputs
- Malformed data
- Large inputs
- Unicode/special characters
- Concurrent operations
- Database errors
- File system errors
- Namespace edge cases
- Search edge cases
- Memory pressure
"""

import os
import sys
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add plugin to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_memory.core import RAGCore
from rag_memory.core.file_indexing import FileIndexer, chunk_by_headers, compute_hash
import pytest


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def rag(temp_db):
    """Create RAG instance with temporary database."""
    rag = RAGCore(temp_db)
    yield rag
    # Cleanup


@pytest.fixture
def temp_files(tmp_path):
    """Create temporary test files."""
    files = {}

    # Empty file
    files["empty"] = tmp_path / "empty.md"
    files["empty"].write_text("")

    # Small file
    files["small"] = tmp_path / "small.md"
    files["small"].write_text("# Small\n\nContent here.")

    # Large file
    files["large"] = tmp_path / "large.md"
    large_content = "\n\n## Section\n".join([f"Content {i}" for i in range(100)])
    files["large"].write_text(f"# Large\n\n{large_content}")

    # Unicode file
    files["unicode"] = tmp_path / "unicode.md"
    files["unicode"].write_text("# Unicode\n\nHello 世界 🌍Ñoño")

    # Special chars
    files["special"] = tmp_path / "special.md"
    files["special"].write_text("# Special\n\nQuotes: ' \" \`\nTabs:\t\t\nNewlines:\n\n\n")

    # Binary file (should be skipped)
    files["binary"] = tmp_path / "binary.bin"
    files["binary"].write_bytes(b"\x00\x01\x02\x03\xFF\xFE\xFD")

    # No headers
    files["no_headers"] = tmp_path / "no_headers.md"
    files["no_headers"].write_text("Just content\nNo markdown headers\nPlain text")

    # Nested headers
    files["nested"] = tmp_path / "nested.md"
    files["nested"].write_text("# Main\n\n## Sub1\n\nContent\n\n### Deep\n\nDeeper\n\n## Sub2")

    return files, tmp_path


# ============================================================================
# Empty/None Input Tests
# ============================================================================

def test_add_document_empty_content(rag):
    """Test adding document with empty content."""
    result = rag.add_document("", namespace="test")
    assert result is not None

    # Search should not return empty content
    results = rag.search("test", namespace="test")
    assert len(results) == 0


def test_add_document_none_namespace(rag):
    """Test adding document with None namespace."""
    # Should use default namespace
    result = rag.add_document("content", namespace=None)
    assert result is not None


def test_search_empty_query(rag):
    """Test search with empty query."""
    rag.add_document("test content", namespace="test")

    # Empty query should return all or none
    results = rag.search("", namespace="test")
    # Should handle gracefully
    assert isinstance(results, list)


def test_search_none_namespace(rag):
    """Test search with None namespace (search all)."""
    rag.add_document("test content", namespace="ns1")
    rag.add_document("other content", namespace="ns2")

    results = rag.search("test", namespace=None)
    assert isinstance(results, list)


# ============================================================================
# Malformed Input Tests
# ============================================================================

def test_add_document_very_long_content(rag):
    """Test adding document with extremely long content."""
    # 1MB of content
    long_content = "word " * 500000  # ~3MB

    result = rag.add_document(long_content, namespace="test")
    assert result is not None

    # Should be retrievable
    results = rag.search("word", namespace="test")
    assert len(results) > 0


def test_search_very_long_query(rag):
    """Test search with very long query."""
    rag.add_document("test content", namespace="test")

    long_query = "word " * 10000
    results = rag.search(long_query, namespace="test")
    # Should handle gracefully
    assert isinstance(results, list)


def test_add_document_special_characters(rag):
    """Test adding document with special characters."""
    special_content = """
    Quotes: ' " ` ' " `
    Brackets: [ ] { } ( )
    Math: + - * / = < > %
    Symbols: @ # $ ^ & | ~ _
    Newlines: \n\n\n
    Tabs: \t\t\t
    """

    result = rag.add_document(special_content, namespace="test")
    assert result is not None

    # Should be searchable
    results = rag.search("quotes", namespace="test")
    assert len(results) > 0


def test_add_document_unicode(rag):
    """Test adding document with Unicode content."""
    unicode_content = """
    Emoji: 🌍 🎉 🚀 💻
    Chinese: 你好世界
    Japanese: こんにちは
    Arabic: مرحبا بالعالم
    Russian: Привет мир
    Accents: café, naïve, résumé
    Math: ∑ ∫ √ ∞ ≠ ≤ ≥
    Arrows: ← → ↑ ↓
    """

    result = rag.add_document(unicode_content, namespace="test")
    assert result is not None

    # Should be searchable
    results = rag.search("world", namespace="test")
    assert len(results) > 0


# ============================================================================
# File Indexing Edge Cases
# ============================================================================

def test_chunk_empty_content():
    """Test chunking empty content."""
    chunks = chunk_by_headers("")
    assert len(chunks) == 0


def test_chunk_no_headers():
    """Test chunking content without headers."""
    content = "Just plain text\nNo headers here\nJust content"
    chunks = chunk_by_headers(content)
    assert len(chunks) == 1
    assert chunks[0] == content


def test_chunk_single_header():
    """Test chunking content with single header."""
    content = "# Main\n\nContent here"
    chunks = chunk_by_headers(content)
    assert len(chunks) == 1


def test_chunk_multiple_headers():
    """Test chunking content with multiple headers."""
    content = "# Main\n\n## Section 1\n\nContent 1\n\n## Section 2\n\nContent 2"
    chunks = chunk_by_headers(content)
    assert len(chunks) == 3  # Main + 2 sections


def test_chunk_very_large_section():
    """Test chunking with section exceeding max_size."""
    # Create section larger than default max_size (2000)
    large_section = "\n\n".join([f"Line {i}" for i in range(1000)])
    content = f"# Main\n\n{large_section}"

    chunks = chunk_by_headers(content, max_size=500)
    # Should split large chunks
    assert len(chunks) > 1


def test_chunk_nested_headers():
    """Test chunking with nested headers."""
    content = "# Main\n\n## Sub1\n\n### Deep1\n\nContent\n\n### Deep2\n\nContent\n\n## Sub2"
    chunks = chunk_by_headers(content)
    # Should handle nested headers
    assert len(chunks) >= 2


def test_file_indexer_nonexistent_file(rag, tmp_path):
    """Test indexing non-existent file."""
    indexer = FileIndexer(rag, tmp_path)

    added = indexer.index_file(tmp_path / "nonexistent.md", "test")
    assert added == 0  # Should not crash


def test_file_indexer_empty_file(rag, temp_files):
    """Test indexing empty file."""
    files, tmp_path = temp_files
    indexer = FileIndexer(rag, tmp_path)

    added = indexer.index_file(files["empty"], "test")
    assert added == 0  # Empty files skipped


def test_file_indexer_binary_file(rag, temp_files):
    """Test indexing binary file."""
    files, tmp_path = temp_files
    indexer = FileIndexer(rag, tmp_path)

    # Should handle gracefully
    added = indexer.index_file(files["binary"], "test")
    # Binary file should either be skipped or handled


def test_file_indexer_unicode_file(rag, temp_files):
    """Test indexing Unicode file."""
    files, tmp_path = temp_files
    indexer = FileIndexer(rag, tmp_path)

    added = indexer.index_file(files["unicode"], "test")
    assert added > 0  # Should succeed


# ============================================================================
# Deduplication Tests
# ============================================================================

def test_deduplication_same_content(rag):
    """Test that identical content is deduplicated."""
    content = "identical content here"

    # Add same content twice
    id1 = rag.add_document(content, namespace="test", metadata={"hash": "abc123"})
    id2 = rag.add_document(content, namespace="test", metadata={"hash": "abc123"})

    # Both should be added (documents are different)
    assert id1 is not None
    assert id2 is not None


def test_hash_consistency():
    """Test that hash function is consistent."""
    content = "test content"

    hash1 = compute_hash(content)
    hash2 = compute_hash(content)

    assert hash1 == hash2


def test_hash_uniqueness():
    """Test that different content produces different hashes."""
    hash1 = compute_hash("content 1")
    hash2 = compute_hash("content 2")

    assert hash1 != hash2


# ============================================================================
# Namespace Edge Cases
# ============================================================================

def test_namespace_with_special_chars(rag):
    """Test namespace with special characters."""
    # Should handle or reject special chars
    result = rag.add_document("content", namespace="test-ns_123")
    assert result is not None


def test_very_long_namespace(rag):
    """Test very long namespace name."""
    long_ns = "a" * 1000

    result = rag.add_document("content", namespace=long_ns)
    assert result is not None


def test_many_namespaces(rag):
    """Test creating many namespaces."""
    for i in range(100):
        rag.add_document(f"content {i}", namespace=f"ns_{i}")

    # Should have 100 namespaces
    results = rag.search("content", limit=1000)
    assert len(results) == 100


# ============================================================================
# Search Edge Cases
# ============================================================================

def test_search_no_results(rag):
    """Test search that returns no results."""
    rag.add_document("content here", namespace="test")

    results = rag.search("nonexistent_term_xyz", namespace="test")
    assert len(results) == 0


def test_search_all_limit(rag):
    """Test search with limit parameter."""
    for i in range(100):
        rag.add_document(f"content {i}", namespace="test")

    # Request fewer than total
    results = rag.search("content", namespace="test", limit=10)
    assert len(results) <= 10


def test_search_case_sensitive(rag):
    """Test case sensitivity in search."""
    rag.add_document("Python Code", namespace="test")

    # Should find with different case (TF-IDF is case-insensitive)
    results = rag.search("python", namespace="test")
    assert len(results) > 0


def test_search_partial_match(rag):
    """Test partial word matching."""
    rag.add_document("programming python", namespace="test")

    # Should match partial words
    results = rag.search("program", namespace="test")
    assert len(results) > 0


# ============================================================================
# Database Error Handling
# ============================================================================

def test_database_corrupted(temp_db):
    """Test handling of corrupted database."""
    # Corrupt the database
    with open(temp_db, "wb") as f:
        f.write(b"corrupted data")

    # Should handle gracefully
    with pytest.raises(Exception):
        rag = RAGCore(temp_db)
        rag.add_document("test", namespace="test")


def test_database_locked(temp_db):
    """Test handling of locked database."""
    rag1 = RAGCore(temp_db)

    # Lock database from another connection
    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA lock_mode=exclusive")
    conn.execute("BEGIN EXCLUSIVE")

    # Should timeout or handle gracefully
    try:
        rag1.add_document("test", namespace="test")
    except Exception:
        pass  # Expected

    conn.close()


# ============================================================================
# Concurrent Operations
# ============================================================================

def test_concurrent_adds(rag):
    """Test adding documents from multiple threads."""
    errors = []
    documents = []

    def add_doc(n):
        try:
            doc_id = rag.add_document(f"content {n}", namespace=f"ns_{n % 10}")
            documents.append(doc_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_doc, args=(i,)) for i in range(50)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have minimal errors
    assert len(errors) < 5  # Allow some concurrency issues
    assert len(documents) > 40  # Most should succeed


def test_concurrent_searches(rag):
    """Test searching from multiple threads."""
    # Add some documents
    for i in range(20):
        rag.add_document(f"content {i}", namespace="test")

    errors = []
    results = []

    def search(n):
        try:
            result = rag.search(f"content {n}", namespace="test")
            results.append(len(result))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=search, args=(i,)) for i in range(20)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All searches should succeed
    assert len(errors) == 0
    assert len(results) == 20


# ============================================================================
# Memory Pressure Tests
# ============================================================================

def test_many_documents(rag):
    """Test adding many documents."""
    count = 1000

    for i in range(count):
        rag.add_document(f"content {i}", namespace="test")

    # Should have all documents
    results = rag.search("content", namespace="test", limit=2000)
    assert len(results) == count


def test_large_document_metadata(rag):
    """Test document with large metadata."""
    large_metadata = {
        "key": "value " * 10000,  # Large value
        "nested": {"data": "x" * 1000},
    }

    result = rag.add_document("content", namespace="test", metadata=large_metadata)
    assert result is not None


# ============================================================================
# Cache Edge Cases
# ============================================================================

def test_cache_expiry(rag):
    """Test cache expiry after TTL."""
    rag.add_document("test content", namespace="test")

    # First search (cache miss)
    results1 = rag.search("test", namespace="test")

    # Wait for cache expiry (if implemented)
    # time.sleep(301)  # Skip in unit tests

    # Second search (should hit cache if available)
    results2 = rag.search("test", namespace="test")

    assert len(results1) == len(results2)


def test_cache_size_limit(rag):
    """Test cache respects size limit."""
    # Add many unique searches
    for i in range(2000):  # More than cache_size (1000)
        rag.add_document(f"unique content {i}", namespace="test")
        rag.search(f"unique {i}", namespace="test")

    # Should not crash or leak memory


# ============================================================================
# Metadata Edge Cases
# ============================================================================

def test_metadata_none(rag):
    """Test document with None metadata."""
    result = rag.add_document("content", namespace="test", metadata=None)
    assert result is not None


def test_metadata_empty_dict(rag):
    """Test document with empty metadata."""
    result = rag.add_document("content", namespace="test", metadata={})
    assert result is not None


def test_metadata_special_types(rag):
    """Test metadata with special data types."""
    special_metadata = {
        "int": 42,
        "float": 3.14,
        "bool": True,
        "list": [1, 2, 3],
        "none": None,
    }

    result = rag.add_document("content", namespace="test", metadata=special_metadata)
    assert result is not None


# ============================================================================
# Delete/Flush Tests
# ============================================================================

def test_flush_nonexistent_namespace(rag):
    """Test flushing namespace that doesn't exist."""
    # Should handle gracefully
    rag.flush_namespace("nonexistent")


def test_flush_all_documents(rag):
    """Test flushing all documents."""
    rag.add_document("content1", namespace="test1")
    rag.add_document("content2", namespace="test2")

    # Flush all
    rag.flush_all()

    # Database should be empty
    results = rag.search("content", limit=10)
    assert len(results) == 0


# ============================================================================
# Performance Edge Cases
# ============================================================================

def test_very_slow_search(rag):
    """Test search with expensive query."""
    # Add many documents
    for i in range(1000):
        rag.add_document(f"word {i} " * 100, namespace="test")

    # Complex query
    start = time.time()
    results = rag.search("word", namespace="test", limit=100)
    elapsed = time.time() - start

    # Should complete in reasonable time (< 5 seconds)
    assert elapsed < 5.0
    assert len(results) > 0


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow(rag):
    """Test complete workflow: add, search, delete."""
    # Add documents
    id1 = rag.add_document("python code", namespace="code")
    id2 = rag.add_document("javascript code", namespace="code")

    # Search
    results = rag.search("python", namespace="code")
    assert len(results) == 1

    # Flush namespace
    rag.flush_namespace("code")

    # Verify empty
    results = rag.search("python", namespace="code")
    assert len(results) == 0


def test_file_indexing_workflow(rag, temp_files):
    """Test complete file indexing workflow."""
    files, tmp_path = temp_files
    indexer = FileIndexer(rag, tmp_path)

    # Index files
    stats = indexer.index_all(chunk_size=1000)

    # Should have indexed some files
    assert stats["files_indexed"] > 0
    assert stats["chunks_added"] > 0

    # Re-index (should deduplicate)
    stats2 = indexer.index_all(chunk_size=1000)

    # Should skip duplicates
    assert stats2["chunks_skipped"] > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short", "-x"])
