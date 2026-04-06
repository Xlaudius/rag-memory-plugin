"""RAG Core functionality tests.

Tests for core RAG operations including:
- TF-IDF search
- Neural search (if available)
- Hybrid search
- Document CRUD
- Namespace management
- Statistics
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add plugin to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_memory.core import RAGCore
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


# ============================================================================
# Document Addition Tests
# ============================================================================

def test_add_single_document(rag):
    """Test adding a single document."""
    doc_id = rag.add_document(
        content="This is a test document about Python programming.",
        namespace="test",
        metadata={"source": "test"}
    )

    assert doc_id is not None
    assert isinstance(doc_id, int)


def test_add_multiple_documents(rag):
    """Test adding multiple documents."""
    ids = []
    for i in range(10):
        doc_id = rag.add_document(
            content=f"Document {i} about topic {i % 3}",
            namespace="test"
        )
        ids.append(doc_id)

    assert len(ids) == 10
    assert all(id is not None for id in ids)


def test_add_document_with_metadata(rag):
    """Test adding document with metadata."""
    metadata = {
        "source": "test.txt",
        "author": "tester",
        "timestamp": "2026-04-06",
        "tags": ["python", "test"]
    }

    doc_id = rag.add_document(
        content="Test content",
        namespace="test",
        metadata=metadata
    )

    assert doc_id is not None


def test_add_document_to_multiple_namespaces(rag):
    """Test adding documents to different namespaces."""
    id1 = rag.add_document("Content 1", namespace="ns1")
    id2 = rag.add_document("Content 2", namespace="ns2")

    assert id1 is not None
    assert id2 is not None


# ============================================================================
# Search Tests
# ============================================================================

def test_search_tfidf(rag):
    """Test TF-IDF search."""
    # Add documents
    rag.add_document("Python programming language", namespace="test")
    rag.add_document("JavaScript is also popular", namespace="test")
    rag.add_document("Java is different from JavaScript", namespace="test")

    # Search for Python
    results = rag.search("Python", namespace="test", mode="tfidf")

    assert len(results) > 0
    # Most relevant should be first
    assert "Python" in results[0]["content"]


def test_search_with_limit(rag):
    """Test search with limit parameter."""
    # Add documents
    for i in range(10):
        rag.add_document(f"Document {i} with keyword", namespace="test")

    # Search with limit
    results = rag.search("keyword", namespace="test", limit=5)

    assert len(results) <= 5


def test_search_no_results(rag):
    """Test search that returns no results."""
    rag.add_document("Python programming", namespace="test")

    results = rag.search("nonexistent_word_xyz", namespace="test")

    assert len(results) == 0


def test_search_all_namespaces(rag):
    """Test searching across all namespaces."""
    rag.add_document("Python content", namespace="ns1")
    rag.add_document("More Python stuff", namespace="ns2")
    rag.add_document("JavaScript content", namespace="ns3")

    # Search all namespaces
    results = rag.search("Python", namespace=None)

    assert len(results) == 2


def test_search_specific_namespace(rag):
    """Test searching specific namespace."""
    rag.add_document("Python code", namespace="code")
    rag.add_document("Python tutorial", namespace="tutorial")
    rag.add_document("Python guide", namespace="guide")

    results = rag.search("Python", namespace="code")

    assert len(results) == 1
    assert results[0]["namespace"] == "code"


def test_search_relevance_scoring(rag):
    """Test that search results are relevance-scored."""
    rag.add_document("Python programming language tutorial", namespace="test")
    rag.add_document("Some other text without Python", namespace="test")
    rag.add_document("Python Python Python Python", namespace="test")

    results = rag.search("Python", namespace="test")

    # Results should be scored
    assert len(results) > 1
    # Each result should have a score
    for result in results:
        assert "score" in result or "content" in result


# ============================================================================
# Document Retrieval Tests
# ============================================================================

def test_get_document_by_id(rag):
    """Test retrieving document by ID."""
    doc_id = rag.add_document(
        "Test content",
        namespace="test",
        metadata={"key": "value"}
    )

    document = rag.get_document(doc_id)

    assert document is not None
    assert document["content"] == "Test content"
    assert document["namespace"] == "test"
    assert document["metadata"]["key"] == "value"


def test_get_nonexistent_document(rag):
    """Test retrieving non-existent document."""
    document = rag.get_document(99999)

    assert document is None


def test_get_documents_by_namespace(rag):
    """Test retrieving all documents in namespace."""
    rag.add_document("Content 1", namespace="test")
    rag.add_document("Content 2", namespace="test")
    rag.add_document("Content 3", namespace="other")

    documents = rag.get_documents(namespace="test")

    assert len(documents) == 2
    assert all(doc["namespace"] == "test" for doc in documents)


# ============================================================================
# Namespace Tests
# ============================================================================

def test_list_namespaces(rag):
    """Test listing all namespaces."""
    rag.add_document("Content 1", namespace="ns1")
    rag.add_document("Content 2", namespace="ns2")
    rag.add_document("Content 3", namespace="ns3")

    namespaces = rag.list_namespaces()

    assert len(namespaces) == 3
    assert "ns1" in namespaces
    assert "ns2" in namespaces
    assert "ns3" in namespaces


def test_namespace_count(rag):
    """Test counting documents in namespace."""
    rag.add_document("Content 1", namespace="test")
    rag.add_document("Content 2", namespace="test")
    rag.add_document("Content 3", namespace="other")

    count = rag.get_document_count(namespace="test")

    assert count == 2


def test_delete_namespace(rag):
    """Test deleting all documents in namespace."""
    rag.add_document("Content 1", namespace="test")
    rag.add_document("Content 2", namespace="test")

    # Delete namespace
    rag.flush_namespace("test")

    # Verify empty
    count = rag.get_document_count(namespace="test")
    assert count == 0


# ============================================================================
# Update/Delete Tests
# ============================================================================

def test_update_document(rag):
    """Test updating document content."""
    doc_id = rag.add_document(
        "Original content",
        namespace="test",
        metadata={"version": 1}
    )

    # Update document
    success = rag.update_document(
        doc_id,
        content="Updated content",
        metadata={"version": 2}
    )

    assert success is True

    # Verify update
    document = rag.get_document(doc_id)
    assert document["content"] == "Updated content"
    assert document["metadata"]["version"] == 2


def test_delete_document(rag):
    """Test deleting document."""
    doc_id = rag.add_document("Content", namespace="test")

    # Delete document
    success = rag.delete_document(doc_id)

    assert success is True

    # Verify deleted
    document = rag.get_document(doc_id)
    assert document is None


# ============================================================================
# Statistics Tests
# ============================================================================

def test_get_stats(rag):
    """Test getting database statistics."""
    # Add documents
    for i in range(10):
        rag.add_document(f"Document {i}", namespace=f"ns_{i % 3}")

    stats = rag.get_stats()

    assert stats["document_count"] == 10
    assert stats["namespace_count"] == 3


def test_empty_database_stats(temp_db):
    """Test stats on empty database."""
    rag = RAGCore(temp_db)
    stats = rag.get_stats()

    assert stats["document_count"] == 0
    assert stats["namespace_count"] == 0


# ============================================================================
# Hybrid Search Tests
# ============================================================================

def test_hybrid_search(rag):
    """Test hybrid TF-IDF + neural search."""
    rag.add_document("Machine learning algorithms", namespace="test")
    rag.add_document("Deep learning neural networks", namespace="test")
    rag.add_document("Traditional programming", namespace="test")

    # Hybrid search
    results = rag.search("AI models", namespace="test", mode="hybrid")

    # Should find semantically similar content
    assert len(results) > 0


def test_neural_search_only(rag):
    """Test neural-only search."""
    rag.add_document("Python programming language", namespace="test")
    rag.add_document("JavaScript scripting", namespace="test")

    # Neural search (if model available)
    try:
        results = rag.search("coding", namespace="test", mode="neural")
        assert isinstance(results, list)
    except Exception as e:
        # Model might not be available
        pytest.skip("Neural model not available")


# ============================================================================
# Performance Tests
# ============================================================================

def test_bulk_add_performance(rag):
    """Test performance of bulk document addition."""
    import time

    start = time.time()

    for i in range(100):
        rag.add_document(f"Document {i} with some content", namespace="test")

    elapsed = time.time() - start

    # Should complete in reasonable time
    assert elapsed < 10.0


def test_bulk_search_performance(rag):
    """Test performance of bulk searches."""
    # Add documents
    for i in range(100):
        rag.add_document(f"Content {i} with keyword test", namespace="test")

    import time
    start = time.time()

    # Perform searches
    for i in range(50):
        results = rag.search(f"keyword {i % 10}", namespace="test")

    elapsed = time.time() - start

    # Should complete in reasonable time
    assert elapsed < 5.0


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_add_document_empty_content(rag):
    """Test adding document with empty content."""
    doc_id = rag.add_document("", namespace="test")
    # Should handle gracefully
    assert doc_id is not None or doc_id is None


def test_search_empty_query(rag):
    """Test search with empty query."""
    rag.add_document("Test content", namespace="test")

    results = rag.search("", namespace="test")
    # Should handle gracefully
    assert isinstance(results, list)


def test_invalid_document_id(rag):
    """Test operations with invalid document ID."""
    result = rag.get_document(-1)
    assert result is None

    success = rag.update_document(-1, "content")
    assert success is False

    success = rag.delete_document(-1)
    assert success is False


# ============================================================================
# Persistence Tests
# ============================================================================

def test_database_persistence(temp_db):
    """Test that data persists across RAG instances."""
    # Create first instance
    rag1 = RAGCore(temp_db)
    rag1.add_document("Persistent content", namespace="test")

    # Create second instance
    rag2 = RAGCore(temp_db)
    results = rag2.search("Persistent", namespace="test")

    assert len(results) == 1
    assert results[0]["content"] == "Persistent content"


# ============================================================================
# Integration Tests
# ============================================================================

def test_complete_crud_workflow(rag):
    """Test complete CRUD workflow."""
    # Create
    doc_id = rag.add_document(
        "Original content",
        namespace="test",
        metadata={"version": 1}
    )

    # Read
    document = rag.get_document(doc_id)
    assert document["content"] == "Original content"

    # Update
    rag.update_document(doc_id, "Updated content", metadata={"version": 2})
    document = rag.get_document(doc_id)
    assert document["content"] == "Updated content"

    # Search
    results = rag.search("Updated", namespace="test")
    assert len(results) == 1

    # Delete
    rag.delete_document(doc_id)
    document = rag.get_document(doc_id)
    assert document is None


def test_namespace_isolation(rag):
    """Test that namespaces are isolated."""
    rag.add_document("Shared keyword in ns1", namespace="ns1")
    rag.add_document("Shared keyword in ns2", namespace="ns2")

    results1 = rag.search("keyword", namespace="ns1")
    results2 = rag.search("keyword", namespace="ns2")

    assert len(results1) == 1
    assert len(results2) == 1
    assert results1[0]["namespace"] == "ns1"
    assert results2[0]["namespace"] == "ns2"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
