#!/usr/bin/env python3
"""
StudyForge AI - Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="studyforge-ai",
    version="1.0.0",
    author="StudyForge AI Team",
    author_email="team@studyforge-ai.com",
    description="Enterprise-grade AI study assistant with bulletproof timeout management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/StudyForge-AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "studyforge=main:main",
            "studyforge-ai=main:main",
            "studyforge-test=tests.timeout_functionality_test:main",
        ],
    },
    keywords=[
        "ai", "education", "study", "assistant", "timeout", "ollama",
        "machine-learning", "artificial-intelligence", "student", "learning",
        "enterprise", "reliability", "timeout-management"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/StudyForge-AI/issues",
        "Source": "https://github.com/yourusername/StudyForge-AI",
        "Documentation": "https://github.com/yourusername/StudyForge-AI/tree/main/docs",
        "Homepage": "https://studyforge-ai.com",
    },
    include_package_data=True,
    zip_safe=False,
)