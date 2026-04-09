"""Test suite for RAG Memory Plugin."""

import pytest
from pathlib import Path


def test_readme_exists():
    """Test that README.md exists."""
    readme_path = Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md not found"


def test_migration_guide_exists():
    """Test that MIGRATION.md exists."""
    migration_path = Path(__file__).parent.parent / "MIGRATION.md"
    assert migration_path.exists(), "MIGRATION.md not found"


def test_install_script_exists():
    """Test that install.sh exists."""
    install_script = Path(__file__).parent.parent / "install.sh"
    assert install_script.exists(), "install.sh not found"
    assert install_script.stat().st_mode & 0o111, "install.sh is not executable"


def test_migration_script_exists():
    """Test that migrate_to_new_venv.sh exists."""
    migration_script = Path(__file__).parent.parent / "migrate_to_new_venv.sh"
    assert migration_script.exists(), "migrate_to_new_venv.sh not found"
    assert migration_script.stat().st_mode & 0o111, "migrate_to_new_venv.sh is not executable"


def test_bash_syntax():
    """Test that bash scripts have valid syntax."""
    import subprocess

    base_path = Path(__file__).parent.parent

    scripts = [
        base_path / "install.sh",
        base_path / "migrate_to_new_venv.sh",
    ]

    for script in scripts:
        if script.exists():
            result = subprocess.run(
                ["bash", "-n", str(script)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"Syntax error in {script.name}: {result.stderr}"


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"


def test_readme_has_installation_instructions():
    """Test that README.md has installation instructions."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    assert "curl -sSL" in content, "README.md missing installation script link"
    assert "rag-memory setup" in content, "README.md missing setup instructions"
    assert "rag-memory doctor" in content, "README.md missing doctor instructions"


def test_readme_has_migration_section():
    """Test that README.md has migration section."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    assert "Migration from Old Installation" in content, "README.md missing migration section"
    assert "MIGRATION.md" in content, "README.md not linking to MIGRATION.md"


def test_gitignore_exists():
    """Test that .gitignore exists."""
    gitignore_path = Path(__file__).parent.parent / ".gitignore"
    assert gitignore_path.exists(), ".gitignore not found"


def test_license_exists():
    """Test that LICENSE file exists."""
    license_path = Path(__file__).parent.parent / "LICENSE"
    assert license_path.exists(), "LICENSE not found"


@pytest.mark.benchmark
def test_documentation_structure():
    """Test documentation structure (benchmark)."""
    base_path = Path(__file__).parent.parent

    required_files = [
        "README.md",
        "MIGRATION.md",
        "LICENSE",
        "pyproject.toml",
        "install.sh",
    ]

    for filename in required_files:
        file_path = base_path / filename
        assert file_path.exists(), f"Required file {filename} not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
