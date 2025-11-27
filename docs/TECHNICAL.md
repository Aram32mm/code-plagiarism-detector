# Technical Documentation

A detailed explanation of the dual-analysis approach used for code plagiarism detection.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Structural Analysis](#structural-analysis)
3. [Syntactic Analysis](#syntactic-analysis)
4. [Why Both Methods Are Necessary](#why-both-methods-are-necessary)
5. [Implementation Details](#implementation-details)
6. [Limitations and Edge Cases](#limitations-and-edge-cases)

---

## Problem Statement

Code plagiarism detection faces several challenges that simple text comparison cannot address:

- **Variable renaming**: Changing `count` to `total` should not evade detection
- **Reformatting**: Adding whitespace or restructuring blocks preserves logic
- **Comment modification**: Removing or rewriting comments is trivial
- **Language translation**: Rewriting Python code in Java maintains the algorithm

A robust detector must identify semantic similarity rather than textual similarity.

---

## Structural Analysis

### Concept

Structural analysis extracts the control flow skeleton of code, reducing it to a sequence of abstract patterns. Two implementations of the same algorithm will share the same control flow structure regardless of variable names, formatting, or even programming language.

### Extraction Process

1. Parse source code into an Abstract Syntax Tree (AST)
2. Traverse the AST and identify control flow nodes
3. Map language-specific nodes to universal patterns:
   - `for_statement`, `while_statement` → `LOOP`
   - `if_statement`, `elif_clause` → `COND`
4. Record the nesting depth relative to other control flow nodes
5. Generate pattern strings: `LOOP:d0`, `COND:d1`, `COND:d2`

### Depth Normalization

The depth value represents the number of control flow ancestors, not the raw AST depth. This normalization ensures:

```
# These produce identical patterns despite different AST depths

def func():              class Solution:
    for i in range(n):       def solve(self):
        if x > 0:                for i in range(n):
            pass                     if x > 0:
                                         pass
```

Both yield: `LOOP:d0`, `COND:d1`

### Else-If Chain Handling

A critical distinction exists between else-if chains and nested conditionals:

```cpp
// Else-if chain (siblings at same depth)
if (a) {
    // ...
} else if (b) {
    // ...
}
// Pattern: COND:d0, COND:d0

// Nested conditional (parent-child relationship)
if (a) {
    if (b) {
        // ...
    }
}
// Pattern: COND:d0, COND:d1
```

The detector handles this by treating `else_clause` (C++), `elif_clause` (Python), and `else` tokens followed by `if_statement` (Java) as passthrough nodes that do not increment depth.

### Comparison Method

Patterns are compared using Jaccard similarity over 2-shingles:

1. Generate all consecutive pairs of patterns
2. Compute intersection and union of shingle sets
3. Similarity = |intersection| / |union|

### Strengths

- Language-agnostic comparison
- Resistant to renaming and reformatting
- Captures algorithmic structure
- Effective for cross-language plagiarism

### Weaknesses

- Coarse granularity loses fine details
- Similar algorithms produce similar patterns regardless of origin
- Cannot distinguish between independent implementations of standard algorithms
- Small code snippets may lack sufficient patterns for reliable comparison

---

## Syntactic Analysis

### Concept

Syntactic analysis generates a perceptual hash of the code structure. Unlike cryptographic hashes, perceptual hashes produce similar outputs for similar inputs, enabling similarity measurement via Hamming distance.

### Hash Generation Process

1. Parse source code into an AST
2. Extract structural features with depth information:
   - `function_definition:d0`
   - `for_statement:d1`
   - `if_statement:d2`
3. Normalize features to universal patterns where applicable
4. Generate k-shingles (k adapts based on feature count)
5. Apply Locality-Sensitive Hashing (LSH) to produce a 256-bit hash

### Locality-Sensitive Hashing

LSH ensures that similar feature sets produce similar hashes:

```
hash_bits = []
for i in range(32):
    h = SHA256("salt" + i)
    for shingle in shingles:
        h.update(shingle)
    hash_bits.append(h.digest()[0])

return unpack_bits(hash_bits)  # 256-bit array
```

### Comparison Method

Hashes are compared using Hamming distance:

```
similarity = 1.0 - (hamming_distance / 256)
```

### Strengths

- Captures more structural detail than pattern-only analysis
- Fast comparison (single distance calculation)
- Scales to large codebases via database indexing
- Effective for same-language plagiarism with renaming

### Weaknesses

- Language-specific features reduce cross-language accuracy
- Hash collisions possible for very different code
- Less interpretable than pattern matching
- Sensitive to structural reorganization

---

## Why Both Methods Are Necessary

Neither method alone provides comprehensive detection. Their complementary nature addresses different plagiarism scenarios.

### Scenario Matrix

| Scenario                          | Structural | Syntactic | Combined |
|-----------------------------------|------------|-----------|----------|
| Same code, variables renamed      | High       | High      | High     |
| Same algorithm, different style   | High       | Medium    | High     |
| Cross-language translation        | High       | Low       | High     |
| Similar but distinct algorithms   | Medium     | Low       | Medium   |
| Copy-paste with minor edits       | High       | High      | High     |
| Completely different code         | Low        | Low       | Low      |

### Cross-Language Example

Consider bubble sort in Python and C++:

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
```

```cpp
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
```

| Method     | Result                           |
|------------|----------------------------------|
| Structural | `LOOP:d0, LOOP:d1, COND:d2` (identical) |
| Syntactic  | Different hashes (language-specific AST) |
| Combined   | High similarity via structural match |

### Same-Language Renaming Example

```python
# Original
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# Plagiarized (renamed)
def compute_total(values):
    result = 0
    for val in values:
        result += val
    return result
```

| Method     | Result                           |
|------------|----------------------------------|
| Structural | `LOOP:d0` (identical but minimal) |
| Syntactic  | Similar hashes (same structure)  |
| Combined   | High similarity via syntactic match |

### Decision Logic

The final similarity score uses the maximum of both methods:

```python
similarity = max(syntactic_similarity, structural_similarity)
```

This ensures detection regardless of which method is more effective for the specific case.

---

## Implementation Details

### AST Parsing

The system uses tree-sitter for parsing, providing:

- Consistent AST structure across languages
- Incremental parsing capability
- Error-tolerant parsing of incomplete code

### Pattern Mapping

Each language defines mappings from AST node types to universal patterns:

```python
'python': {
    'LOOP': ['for_statement', 'while_statement'],
    'COND': ['if_statement'],
}

'java': {
    'LOOP': ['for_statement', 'while_statement', 'enhanced_for_statement', 'do_statement'],
    'COND': ['if_statement'],
}

'cpp': {
    'LOOP': ['for_statement', 'while_statement', 'for_range_loop', 'do_statement'],
    'COND': ['if_statement'],
}
```

### Database Storage

The reference database stores three representations for each file:

| Field    | Purpose                              |
|----------|--------------------------------------|
| code     | Original source for display         |
| hash     | 256-bit LSH for syntactic search    |
| patterns | Control flow patterns for structural search |

### Search Methods

Three search strategies are available:

1. **Syntactic only**: Fast hash-based search using Hamming distance
2. **Structural only**: Pattern-based search using Jaccard similarity
3. **Combined**: Maximum similarity from both methods (recommended)

---

## Limitations and Edge Cases

### False Positives

Canonical algorithms with identical control flow will match regardless of independent authorship:

- Binary search implementations
- Tree traversals (preorder, inorder, postorder)
- Standard sorting algorithms
- Common design patterns

Mitigation: Use higher thresholds (80%+) and require manual review.

### False Negatives

Significant algorithmic restructuring may evade detection:

- Recursion converted to iteration
- Loop unrolling
- Conditional flattening
- Algorithm substitution (quicksort vs mergesort)

Mitigation: Combine with other detection methods (token-based, semantic).

### Minimum Code Size

Very short snippets produce few patterns, reducing reliability:

- Single-loop code may match many unrelated implementations
- Functions without control flow produce empty patterns

Mitigation: Require minimum pattern count for high-confidence results.

### Language Coverage

Currently limited to Python, Java, and C++. Adding languages requires:

1. tree-sitter grammar installation
2. Pattern mapping configuration
3. Passthrough node identification for else-if handling

---

## References

- Schleimer, S., Wilkerson, D. S., & Aiken, A. (2003). Winnowing: Local algorithms for document fingerprinting.
- Prechelt, L., Malpohl, G., & Philippsen, M. (2002). Finding plagiarisms among a set of programs with JPlag.
- tree-sitter documentation: https://tree-sitter.github.io/tree-sitter/