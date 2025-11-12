# Code Plagiarism Detector

Lightweight code plagiarism detection using perceptual hashing and AST fingerprinting.

## Features

- **AST-based Analysis**: Uses tree-sitter to parse Python, Java, and C++ code into Abstract Syntax Trees
- **Structural Fingerprinting**: Extracts key structural features including:
  - Function and method definitions
  - Control flow structures (if, for, while, try/catch, etc.)
  - Import statements
  - Overall code structure
- **Perceptual Hashing**: Generates 256-bit hashes using Locality Sensitive Hashing (LSH)
- **Similarity Detection**: Compares code using Hamming distance to detect plagiarism
- **Cross-language Support**: Can detect similar algorithms across Python, Java, and C++

## Installation

```bash
# Clone the repository
git clone https://github.com/Aram32mm/code-plagiarism-detector.git
cd code-plagiarism-detector

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from code_plagiarism_detector import CodeHasher

# Initialize the hasher
hasher = CodeHasher()

# Hash a code string
code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

hash1 = hasher.hash_code(code, 'python')

# Hash a file
hash2 = hasher.hash_file('path/to/file.py')

# Compare two hashes
similarity, hamming_distance = hasher.compare(hash1, hash2)
print(f"Similarity: {similarity:.2%}")
print(f"Hamming Distance: {hamming_distance}/256")
```

## API Reference

### CodeHasher Class

#### Methods

- **`hash_code(code: str, language: str) -> np.ndarray`**
  - Generate a 256-bit hash for a code string
  - Args:
    - `code`: Source code string
    - `language`: Programming language ('python', 'java', 'cpp')
  - Returns: 256-bit hash as numpy array

- **`hash_file(file_path: str) -> np.ndarray`**
  - Generate a 256-bit hash for a code file
  - Args:
    - `file_path`: Path to source code file
  - Returns: 256-bit hash as numpy array
  - Automatically detects language from file extension

- **`compare(hash1: np.ndarray, hash2: np.ndarray) -> Tuple[float, int]`**
  - Compare two hashes using Hamming distance
  - Args:
    - `hash1`: First hash
    - `hash2`: Second hash
  - Returns: Tuple of (similarity_score, hamming_distance)
    - `similarity_score`: 0.0 to 1.0 (1.0 = identical)
    - `hamming_distance`: Number of differing bits (0-256)

## How It Works

1. **AST Parsing**: Code is parsed into an Abstract Syntax Tree using tree-sitter
2. **Feature Extraction**: Structural features are extracted from the AST:
   - Function/method definitions
   - Control flow statements
   - Import declarations
   - General structure nodes
3. **Hash Generation**: Features are processed using LSH to create a 256-bit perceptual hash
4. **Comparison**: Hashes are compared using Hamming distance to measure similarity

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/code_plagiarism_detector

# Run specific test
pytest tests/test_code_hasher.py::TestCodeHasher::test_hash_code_python
```

## Examples

See the `examples/` directory for detailed usage examples:

```bash
python examples/basic_usage.py
```

## Supported Languages

- Python (`.py`)
- Java (`.java`)
- C++ (`.cpp`, `.cc`, `.cxx`, `.c`, `.h`, `.hpp`)

## Use Cases

- **Academic Integrity**: Detect plagiarism in programming assignments
- **Code Review**: Identify duplicate code across projects
- **License Compliance**: Find similar code that may have licensing implications
- **Refactoring**: Discover code clones that could be consolidated

## Limitations

- Detection is based on structural similarity, not semantic equivalence
- Heavily obfuscated code may evade detection
- Very small code snippets may not have enough structure for reliable comparison

## License

See LICENSE file for details.
