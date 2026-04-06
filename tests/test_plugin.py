"""Test RAG Memory plugin registration."""

from pathlib import Path


def test_plugin_import():
    """Test that plugin can be imported."""
    try:
        import rag_memory
        assert rag_memory.__version__ == "1.0.0"
        assert hasattr(rag_memory, "register")
        assert rag_memory.plugin_name == "rag-memory"
    except ImportError as e:
        raise AssertionError(f"Failed to import plugin: {e}")


def test_plugin_register_signature():
    """Test that register() has correct signature."""
    import rag_memory
    import inspect

    assert callable(rag_memory.register)
    sig = inspect.signature(rag_memory.register)
    params = list(sig.parameters.keys())
    assert params == ["context"]


def test_entry_point():
    """Test that entry point is configured."""
    import importlib.metadata

    entry_points = list(importlib.metadata.entry_points(group="hermes_agent.plugins"))
    assert any(ep.name == "rag-memory" for ep in entry_points)


def test_cli_import():
    """Test that CLI can be imported."""
    try:
        from rag_memory.cli import main
        assert callable(main)
    except ImportError as e:
        raise AssertionError(f"Failed to import CLI: {e}")


def test_package_structure():
    """Test that package structure is correct."""
    import rag_memory

    pkg_dir = Path(rag_memory.__file__).parent

    # Check core modules exist
    assert (pkg_dir / "core").exists()
    assert (pkg_dir / "tools").exists()
    assert (pkg_dir / "scripts").exists()
