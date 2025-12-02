# Findings and Practical Behavior

This document summarizes the **empirical behavior**, **strengths**, **limitations**, and **recommended usage** of the plagiarism detector implemented in this repository. While the Technical Documentation explains the dual-analysis approach conceptually, this file focuses on how the detector actually behaves **in practice**.

---

## 1. Overall Behavior

The detector combines two complementary similarity measures:

- **Syntactic similarity**  
  A 256-bit perceptual hash derived from normalized AST features. Similar hashes (small Hamming distance) indicate structurally similar code, especially within the same language.

- **Structural similarity**  
  A Jaccard-style similarity over control-flow patterns (e.g. `LOOP:d0`, `COND:d1`), extracted from the AST with language-specific handling of `if`, `else`, `elif`, and related constructs.

A `ComparisonResult` aggregates this information:

- `similarity` – overall similarity (max of structural and syntactic)
- `syntactic_similarity` – hash-based similarity
- `structural_similarity` – pattern-based similarity
- `hamming_distance` – raw distance between hashes
- `matching_patterns` + `pattern_match_ratio` – overlapping control-flow fragments
- `confidence` – `high`, `medium`, or `low`
- `plagiarism_detected` – boolean flag based on configurable thresholds

In practice, the detector tends to:

- Flag **copied or lightly edited code** with high similarity and high confidence.
- Remain **robust to superficial modifications** (renaming, comments, formatting).
- Successfully capture **cross-language** similarities for many common algorithms.
- Require **careful interpretation** on canonical tasks and very short snippets.

---

## 2. Same-Language Behavior

### 2.1 Identical or near-identical code

When two submissions are essentially the same (possibly with minor cosmetic changes):

- The hashes are almost identical (`hamming_distance ≈ 0`).
- `syntactic_similarity` and `structural_similarity` are both very high (often 1.0).
- `similarity` is 1.0 or very close.
- `plagiarism_detected` is set to `True` with `confidence = 'high'`.

Typical case: same function bodies, same control flow, similar AST structure.  
Conclusion: the detector is **very sensitive** to direct copying.

### 2.2 Renamed identifiers

Renaming variables, function names, or parameters (e.g. `arr → items`, `i → x`) has little effect:

- Control-flow patterns (`LOOP`, `COND`, nesting depths) remain unchanged.
- AST structure remains nearly identical, changing only leaf identifiers.
- The hash changes slightly, but similarity remains high.

Conclusion: **simple renaming does not bypass** the detector.

### 2.3 Different algorithms in the same language

When two pieces of code implement genuinely different algorithms (e.g. bubble sort vs quick sort, sort vs binary search, iterative vs recursive factorial):

- Control-flow patterns differ:  
  nested loops vs loop+while; different nesting shapes; recursion vs iteration.
- Structural similarity drops significantly (often below medium thresholds).
- Overall similarity tends to be moderate or low.
- Confidence is `low` or at most `medium`; plagiarism is usually not flagged.

Conclusion: the detector **distinguishes** between different algorithmic structures in the same language.

---

## 3. Cross-Language Behavior

A central feature is **cross-language detection** for Python, Java, and C++.

### 3.1 Classic algorithms (bubble sort, binary search, etc.)

For the same algorithm implemented in different languages (Python, Java, C++):

- The control-flow skeleton is nearly identical:
  - nested loops in bubble sort,
  - loop + inner conditions in binary search,
  - while loops for scanning, etc.
- These constructs map to the same abstract patterns (`LOOP`, `COND`) with similar relative depths.
- Structural similarity is typically very high (often 1.0).
- Confidence is `high`, and plagiarism is flagged when the structure matches closely.

Conclusion: **cross-language translations of algorithms are effectively detected** via structural analysis, even when the syntactic hash differs.

### 3.2 More complex algorithms and data structures

For more realistic examples (merge sort with helper functions, Dijkstra’s algorithm with nested loops and selections, LRU cache state machines):

- The detector tracks complex combinations of loops and conditionals.
- Cross-language versions with similar structure produce high structural similarity.
- Nested loops, sequential decisions, and repeated update patterns all contribute to strong matches.

Conclusion: the structural patterns are expressive enough to capture **real-world algorithmic structure**, not just toy examples.

---

## 4. Control-Flow Pattern Semantics

The detector’s structural behavior is tightly linked to its control-flow extraction.

### 4.1 Loops and nesting

- `for` and `while` constructs are unified under `LOOP`.
- Equivalent single loops in different languages map to the same structural pattern (e.g. `['LOOP:d0']`).
- Double and triple nested loops across languages yield identical or very similar sequences like:
  - `LOOP:d0`, `LOOP:d1`
  - or `LOOP:d0`, `LOOP:d1`, `COND:d2` for patterns with inner conditions.

This means:

- **Loop type** (`for` vs `while`) is intentionally abstracted away.
- **Nesting depth** is preserved in a relative form, so nested loops match nested loops, not flat ones.

### 4.2 If/else and else-if chains

The detector distinguishes:

- **Else-if chains**, where consecutive conditions share the same depth, from
- **Nested conditionals**, where an `if` appears inside the body of another.

Example:

- Else-if chain → `COND:d0, COND:d0`
- Nested `if` → `COND:d0, COND:d1`

Across Python, Java, and C++, constructs like `elif`, `else if`, and `else`+`if` are normalized so that equivalent control-flow chains are treated the same.

Conclusion: the detector captures **subtle differences in branching structure** (chain vs nesting) that are important for algorithmic shape.

---

## 5. Edge Cases and Known Quirks

### 5.1 Empty code

When comparing two empty inputs:

- Structural and syntactic information are both essentially absent.
- The detector treats them as **identical at the structural level**, since both lack any control-flow patterns.

Interpretation:  
Two empty submissions are trivially “similar” to each other. This is logically true but not very meaningful in terms of plagiarism; the context (e.g. both students submitted nothing) matters more than the metric.

### 5.2 Empty vs non-empty

When one side is empty and the other contains code:

- Structural similarity is effectively zero (no shared patterns).
- Overall similarity remains low to moderate at best, depending on the hash.

Interpretation:  
The detector **does not confuse** a non-submission with a genuine solution.

### 5.3 Very short or trivial snippets

For very small pieces of code with **no control flow**:

- The structural side has almost no information (no loops/conditions).
- When both snippets lack control-flow, the structural representation becomes trivial and can appear maximally similar.
- The hash is computed from very few features, making similarity less stable and more sensitive to small changes.

Interpretation:  
For short, trivial snippets, the detector’s results are **not reliable as a plagiarism signal** and should be interpreted cautiously (or ignored).

---

## 6. Strengths and Advantages

From a practical point of view, the detector exhibits several strong properties:

1. **Robustness to superficial modifications**  
   Renaming identifiers, changing whitespace, inserting comments, or reformatting code generally has little impact on similarity scores.

2. **Cross-language capability**  
   Python, Java, and C++ implementations of the same algorithm are often recognized as highly similar thanks to pattern abstraction and depth normalization.

3. **Algorithm-level sensitivity**  
   The focus on loops, conditionals, and nesting patterns makes the detector sensitive to **algorithmic structure**, not just superficial text.

4. **Configurable thresholds and patterns**  
   Thresholds for similarity and confidence (`DetectorConfig`) and language-specific pattern mappings (`LanguageConfig`) can be tuned or extended without altering the core engine.

5. **Explainability via patterns**  
   Extracted patterns and debug information make it possible to see *why* two pieces of code are considered similar, enabling better human judgment.

---

## 7. Limitations and Risk Zones

Despite its strengths, the detector has important limitations:

1. **Canonical algorithm bias**  
   For very standard tasks (bubble sort, binary search, BFS/DFS, etc.), independent solutions often share nearly the same control-flow skeleton. The detector may flag these with **high structural similarity**, even in the absence of actual copying.

2. **Short snippet unreliability**  
   For minimal code fragments:
   - There is too little structure to distinguish between unrelated code.
   - Structural similarity can be artificially high.

3. **Heuristic perceptual hash**  
   The hash is a practical, salted SHA-based fingerprint, not a formal locality-sensitive hash with guaranteed properties. It works well empirically, but remains heuristic.

4. **Static thresholds**  
   Global thresholds for “plagiarism” and “confidence” may not be optimal for every assignment or dataset. Some exercises need stricter thresholds; others need more lenient ones.

5. **Limited semantic understanding**  
   The detector does **not** reason about semantic equivalence beyond structural correlations. Deep refactors or algorithm substitutions may go undetected, and purely structural similarities cannot distinguish shared origin from convergent design.

---

## 8. Recommended Usage Patterns

Given this behavior, the detector is best used with the following guidelines:

- Treat it as a **decision-support tool**, not as an automatic judge.
- Interpret high similarity differently depending on the **type of assignment**:
  - For open-ended tasks, high similarity is genuinely suspicious.
  - For canonical exercises, high structural similarity is partly expected.
- Be especially cautious with:
  - **Very short** submissions,
  - **Highly standard** algorithms,
  - Cases where only one of the two similarity measures (structural vs syntactic) is high.
- Use the exposed patterns and debug information to understand *why* the detector considered two submissions similar, and combine that with contextual information (time of submission, collaboration policies, etc.).
- Adjust thresholds per course or assignment if possible, rather than relying on a single global configuration.

In summary, the detector is effective at highlighting structurally similar code across languages and under superficial obfuscations, but its results must be interpreted through the lens of assignment design, code length, and human judgment.
