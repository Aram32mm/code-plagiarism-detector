"""Setup configuration for code-plagiarism-detector."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="code-plagiarism-detector",
    version="0.1.0",
    author="Aram32mm",
    description="Lightweight code plagiarism detection using perceptual hashing and AST fingerprinting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aram32mm/code-plagiarism-detector",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "tree-sitter>=0.20.0",
        "tree-sitter-python>=0.20.0",
        "tree-sitter-java>=0.20.0",
        "tree-sitter-cpp>=0.20.0",
        "datasketch>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Quality Assurance",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="plagiarism detection code ast perceptual-hashing tree-sitter",
)
