"""Setup configuration for code-plagiarism-detector."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

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
        "tree-sitter>=0.20.0",
        "tree-sitter-python>=0.20.0",
        "tree-sitter-java>=0.20.0",
        "tree-sitter-cpp>=0.20.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="plagiarism detection ast hashing code similarity",
)
