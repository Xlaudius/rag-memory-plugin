"""Quick test runner for RAG Memory plugin.

Runs fast tests without neural model loading.
"""

import os
import sys
from pathlib import Path

# Add plugin to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_memory.core import RAGCore
from rag_memory.core.file_indexing import FileIndexer, chunk_by_headers, compute_hash
import tempfile


def test_chunk_by_headers():
    """Test chunking by headers."""
    print("Testing chunk_by_headers...")

    # Test empty
    chunks = chunk_by_headers("")
    assert len(chunks) == 0

    # Test no headers
    chunks = chunk_by_headers("Just text")
    assert len(chunks) == 1

    # Test with headers
    content = "# Main\n\n## Section 1\n\nContent 1\n\n## Section 2\n\nContent 2"
    chunks = chunk_by_headers(content)
    assert len(chunks) == 3

    print("  ✓ Chunking works correctly")


def test_compute_hash():
    """Test hash computation."""
    print("Testing compute_hash...")

    content = "test content"
    hash1 = compute_hash(content)
    hash2 = compute_hash(content)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256

    # Different content = different hash
    hash3 = compute_hash("different")
    assert hash1 != hash3

    print("  ✓ Hashing works correctly")


def test_add_document():
    """Test adding documents."""
    print("Testing add_document...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        rag = RAGCore(db_path)

        # Add single document
        doc_id = rag.add_document("Test content", namespace="test")
        assert doc_id is not None

        # Add multiple
        for i in range(10):
            doc_id = rag.add_document(f"Content {i}", namespace="test")
            assert doc_id is not None

        print("  ✓ Document addition works")

    finally:
        os.unlink(db_path)


def test_search():
    """Test searching."""
    print("Testing search...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        rag = RAGCore(db_path)

        # Add documents
        rag.add_document("Python programming language", namespace="test")
        rag.add_document("JavaScript is also popular", namespace="test")
        rag.add_document("Java is different", namespace="test")

        # Search
        results = rag.search("Python", namespace="test", mode="tfidf")
        assert len(results) > 0
        assert "Python" in results[0]["content"]

        # Search with limit
        results = rag.search("programming", namespace="test", limit=1)
        assert len(results) <= 1

        # No results
        results = rag.search("nonexistent_xyz", namespace="test")
        assert len(results) == 0

        print("  ✓ Search works correctly")

    finally:
        os.unlink(db_path)


def test_namespaces():
    """Test namespace operations."""
    print("Testing namespaces...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        rag = RAGCore(db_path)

        # Add to different namespaces
        rag.add_document("Content 1", namespace="ns1")
        rag.add_document("Content 2", namespace="ns2")
        rag.add_document("Content 3", namespace="ns1")

        # List namespaces
        namespaces = rag.list_namespaces()
        assert len(namespaces) == 2
        assert "ns1" in namespaces
        assert "ns2" in namespaces

        # Count in namespace
        count = rag.get_document_count(namespace="ns1")
        assert count == 2

        # Search specific namespace
        results = rag.search("Content", namespace="ns1")
        assert len(results) == 2

        print("  ✓ Namespaces work correctly")

    finally:
        os.unlink(db_path)


def test_document_crud():
    """Test document CRUD operations."""
    print("Testing document CRUD...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        rag = RAGCore(db_path)

        # Create
        doc_id = rag.add_document(
            "Original content",
            namespace="test",
            metadata={"version": 1}
        )
        assert doc_id is not None

        # Read
        doc = rag.get_document(doc_id)
        assert doc["content"] == "Original content"
        assert doc["metadata"]["version"] == 1

        # Update
        success = rag.update_document(
            doc_id,
            content="Updated content",
            metadata={"version": 2}
        )
        assert success is True

        doc = rag.get_document(doc_id)
        assert doc["content"] == "Updated content"

        # Delete
        success = rag.delete_document(doc_id)
        assert success is True

        doc = rag.get_document(doc_id)
        assert doc is None

        print("  ✓ Document CRUD works correctly")

    finally:
        os.unlink(db_path)


def test_flush():
    """Test flush operations."""
    print("Testing flush operations...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        rag = RAGCore(db_path)

        # Add documents
        rag.add_document("Content 1", namespace="test")
        rag.add_document("Content 2", namespace="other")

        # Flush namespace
        rag.flush_namespace("test")

        count = rag.get_document_count(namespace="test")
        assert count == 0

        # Flush all
        rag.flush_all()

        namespaces = rag.list_namespaces()
        assert len(namespaces) == 0

        print("  ✓ Flush operations work correctly")

    finally:
        os.unlink(db_path)


def test_stats():
    """Test statistics."""
    print("Testing statistics...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        rag = RAGCore(db_path)

        # Empty stats
        stats = rag.get_stats()
        assert stats["document_count"] == 0

        # Add documents
        for i in range(10):
            rag.add_document(f"Content {i}", namespace=f"ns_{i % 3}")

        stats = rag.get_stats()
        assert stats["document_count"] == 10
        assert stats["namespace_count"] == 3

        print("  ✓ Statistics work correctly")

    finally:
        os.unlink(db_path)


def test_special_characters():
    """Test special characters handling."""
    print("Testing special characters...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        rag = RAGCore(db_path)

        # Unicode
        rag.add_document("Hello 世界 🌍", namespace="test")

        # Special chars
        rag.add_document("Quotes: ' \" `", namespace="test")

        # Newlines/tabs
        rag.add_document("Line1\n\n\tLine2", namespace="test")

        # Search should work
        results = rag.search("world", namespace="test")
        assert len(results) > 0

        print("  ✓ Special characters handled correctly")

    finally:
        os.unlink(db_path)


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        rag = RAGCore(db_path)

        # Empty content
        doc_id = rag.add_document("", namespace="test")
        assert doc_id is not None

        # Very long content
        long_content = "word " * 10000
        doc_id = rag.add_document(long_content, namespace="test")
        assert doc_id is not None

        # Very long query
        long_query = "word " * 1000
        results = rag.search(long_query, namespace="test")
        assert isinstance(results, list)

        # None namespace (should use default)
        doc_id = rag.add_document("content", namespace=None)
        assert doc_id is not None

        print("  ✓ Edge cases handled correctly")

    finally:
        os.unlink(db_path)


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RAG Memory Plugin - Quick Test Suite")
    print("="*60 + "\n")

    tests = [
        test_chunk_by_headers,
        test_compute_hash,
        test_add_document,
        test_search,
        test_namespaces,
        test_document_crud,
        test_flush,
        test_stats,
        test_special_characters,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
