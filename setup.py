from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-memory-plugin",  # PyPI will normalize to rag-memory-plugin
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="Production-grade RAG memory system for Hermes Agent with hybrid TF-IDF + Neural retrieval, auto-capture, and zero-configuration setup",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Clawford Orji",
    python_requires=">=3.10",
    license="MIT",
    keywords=["rag", "retrieval", "memory", "hermes", "agent", "neural-search", "tfidf"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Typing :: Typed",
    ],
    install_requires=[
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "click>=8.0",
        "rich>=13.0",
        "sqlite-vec>=0.1.0",
        "scikit-learn>=0.24.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "neural": [
            "sentence-transformers>=2.2.0",
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=4.0",
            "ruff>=0.4",
            "mypy>=1.8",
        ],
        "all": ["rag-memory-plugin[neural,dev]"],
    },
    entry_points={
        "console_scripts": [
            "rag-memory=rag_memory.cli:main",
        ],
        "hermes_agent.plugins": [
            "rag-memory=rag_memory.plugin",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
