# Code Plagiarism Detector

An AST-based code plagiarism detection system that identifies similar code across Python, Java, and C++. Uses a dual-analysis approach combining structural pattern matching with perceptual hashing to detect plagiarism even when code is renamed, reformatted, or translated between languages.

---

## Features

- **Cross-language detection**: Identifies similar algorithms across Python, Java, and C++
- **Variable renaming resistant**: Detects plagiarism even when identifiers are changed
- **Formatting agnostic**: Ignores whitespace, comments, and code style differences
- **Dual analysis**: Combines structural and syntactic methods for comprehensive detection
- **Reference database**: Build a searchable database of known implementations
- **Batch processing**: Compare multiple files simultaneously
- **Web interface**: Interactive Streamlit UI for easy use

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-plagiarism-detector.git
cd code-plagiarism-detector

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
streamlit>=1.28.0
tree-sitter>=0.21.0
tree-sitter-python>=0.21.0
tree-sitter-java>=0.21.0
tree-sitter-cpp>=0.21.0
numpy>=1.24.0
plotly>=5.18.0
pandas>=2.0.0
```

---

## Usage

### Web Interface

```bash
streamlit run streamlit_app.py
```

### Command Line

```python
from code_plagiarism_detector import CodeHasher

hasher = CodeHasher()

# Compare two code snippets
result = hasher.compare(code1, 'python', code2, 'java')

print(f"Similarity: {result.similarity:.1%}")
print(f"Plagiarism detected: {result.plagiarism_detected}")
print(f"Confidence: {result.confidence}")
```

### Building a Reference Database

```python
from code_plagiarism_detector import CodeHasher, CodeHashDatabase, load_code_bank

hasher = CodeHasher()
db = CodeHashDatabase("reference_hashes.db")

# Load reference implementations from a directory
load_code_bank("./code_bank", hasher, db)

# Search for similar code
query_hash = hasher.hash_code(suspicious_code, 'python')
query_patterns = hasher.debug_patterns(suspicious_code, 'python')['control_flow']
matches = db.find_similar(query_hash, query_patterns, threshold=0.6)
```

---

## Project Structure

```
code-plagiarism-detector/
├── src/
│   └── code_plagiarism_detector/
│       ├── __init__.py
│       ├── code_hasher.py      # Core detection engine
│       └── database.py         # Reference database management
├── tests/
│   ├── test_code_hasher.py     # Unit tests
│   └── test_complex.py         # Integration tests
├── code_bank/                   # Reference implementations
│   ├── python/
│   ├── java/
│   └── cpp/
├── streamlit_app.py            # Web interface
├── requirements.txt
└── README.md
```

---

## How It Works

The detector uses two complementary analysis methods:

### Structural Analysis

Extracts control flow patterns (loops, conditionals) from the AST and compares them using Jaccard similarity. This method excels at cross-language detection because it captures algorithmic structure rather than syntax.

### Syntactic Analysis

Generates a 256-bit perceptual hash from normalized AST features and compares using Hamming distance. This method catches same-language plagiarism where variables are renamed but structure is preserved.

See [TECHNICAL.md](TECHNICAL.md) for detailed explanation.

---

## Confidence Levels

| Level  | Threshold | Interpretation                     |
|--------|-----------|-----------------------------------|
| HIGH   | >= 80%    | Very likely plagiarism            |
| MEDIUM | 60-80%    | Requires manual review            |
| LOW    | < 60%     | Probably original                 |

---

## Configuration

Custom thresholds can be set via `DetectorConfig`:

```python
from code_plagiarism_detector import CodeHasher, DetectorConfig

config = DetectorConfig(
    high_confidence_threshold=0.80,
    medium_confidence_threshold=0.60,
    plagiarism_threshold=0.60,
    hash_bits=256
)

hasher = CodeHasher(config=config)
```

---

## Limitations

- Similar canonical algorithms may produce false positives
- Very short code snippets may yield unreliable results
- Detection indicates similarity, not intent
- Manual review is recommended for all flagged cases

---

## Testing

```bash
pytest tests/ -v
```

---

## License

MIT License

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request