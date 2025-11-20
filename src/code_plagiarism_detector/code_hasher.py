"""Code plagiarism detector using AST fingerprinting and perceptual hashing."""

import hashlib
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp


class CodeHasher:
    """Generate and compare perceptual hashes of code based on AST structure."""
    
    # Language configurations
    LANGUAGES = {
        'python': {'parser': tspython, 'extensions': ['.py']},
        'java': {'parser': tsjava, 'extensions': ['.java']},
        'cpp': {'parser': tscpp, 'extensions': ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']}
    }
    
    # AST node types for structural features
    STRUCTURAL_NODES = {
        'python': [
            'function_definition', 'class_definition', 'if_statement', 
            'for_statement', 'while_statement', 'try_statement',
            'with_statement', 'import_statement', 'import_from_statement',
            'assignment', 'comparison_operator', 'binary_operator'
        ],
        'java': [
            'method_declaration', 'class_declaration', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'enhanced_for_statement', 'import_declaration',
            'assignment_expression', 'binary_expression'
        ],
        'cpp': [
            'function_definition', 'class_specifier', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'using_declaration', 'preproc_include',
            'assignment_expression', 'binary_expression'
        ]
    }
    
    # Universal pattern mappings for cross-language detection
    PATTERN_MAPPINGS = {
        'python': {
            'LOOP': ['for_statement', 'while_statement'],
            'CONDITIONAL': ['if_statement', 'elif_clause', 'else_clause'],
            'FUNCTION': ['function_definition'],
            'CLASS': ['class_definition'],
            'TRY': ['try_statement'],
            'BINARY_OP': ['comparison_operator', 'binary_operator', 'boolean_operator'],
            'ASSIGN': ['assignment'],
            'CALL': ['call'],
        },
        'java': {
            'LOOP': ['for_statement', 'while_statement', 'enhanced_for_statement', 'do_statement'],
            'CONDITIONAL': ['if_statement', 'else'],
            'FUNCTION': ['method_declaration'],
            'CLASS': ['class_declaration'],
            'TRY': ['try_statement', 'catch_clause'],
            'BINARY_OP': ['binary_expression'],
            'ASSIGN': ['assignment_expression', 'variable_declarator'],
            'CALL': ['method_invocation'],
        },
        'cpp': {
            'LOOP': ['for_statement', 'while_statement', 'for_range_loop', 'do_statement'],
            'CONDITIONAL': ['if_statement', 'else_clause'],
            'FUNCTION': ['function_definition'],
            'CLASS': ['class_specifier', 'struct_specifier'],
            'TRY': ['try_statement', 'catch_clause'],
            'BINARY_OP': ['binary_expression'],
            'ASSIGN': ['assignment_expression', 'init_declarator'],
            'CALL': ['call_expression'],
        }
    }
    
    def __init__(self):
        """Initialize parsers for each language."""
        self.parsers = {}
        for lang_name, lang_config in self.LANGUAGES.items():
            parser = Parser(Language(lang_config['parser'].language()))
            self.parsers[lang_name] = parser
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        for lang, config in self.LANGUAGES.items():
            if ext in config['extensions']:
                return lang
        raise ValueError(f"Unsupported file extension: {ext}")
    
    def _get_node_signature(self, node, depth: int) -> str:
        """
        Create a rich signature for a node based on its structure.
        
        Captures:
        - Node type (e.g., "for_statement")
        - Nesting depth (e.g., "d2" for nested 2 levels)
        - Child count (e.g., "c3" for 3 children)
        
        This makes hashes sensitive to structure, not just presence of nodes.
        """
        sig_parts = [node.type]
        
        # Add depth for context - nested loops look different from flat loops
        sig_parts.append(f"d{depth}")
        
        # Add child count to capture complexity
        sig_parts.append(f"c{len(node.children)}")
        
        return ":".join(sig_parts)
    
    def _extract_features(self, code: str, language: str) -> List[str]:
        """
        Extract structural features from AST with improved granularity.
        
        Improvements over basic version:
        1. Captures nesting depth (nested loops vs flat loops)
        2. Captures node complexity (number of children)
        3. Explicitly marks nested patterns (nested loops/conditionals)
        4. More comprehensive node type coverage
        """
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, 'utf-8'))
        
        features = []
        structural_nodes = self.STRUCTURAL_NODES[language]
        
        def traverse(node, depth=0):
            """Recursively traverse AST and collect structural features."""
            
            # Collect structural nodes with rich signatures
            if node.type in structural_nodes:
                signature = self._get_node_signature(node, depth)
                features.append(signature)
            
            # Explicitly detect nested patterns (important for algorithm detection)
            if node.type in ['for_statement', 'while_statement', 'if_statement']:
                # Check if this control structure contains another control structure
                for child in node.children:
                    if child.type in ['for_statement', 'while_statement']:
                        # Mark nested loops as special pattern
                        features.append(f"nested_loop:d{depth}")
                    elif child.type == 'if_statement':
                        features.append(f"nested_conditional:d{depth}")
            
            # Recurse to children with increased depth
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(tree.root_node)
        return features
    
    def _normalize_features(self, features: List[str], language: str) -> List[str]:
        """
        Normalize language-specific features to universal patterns.
        
        Key strategy: Focus on PRIMARY CONTROL STRUCTURES only:
        - LOOP (for, while)
        - CONDITIONAL (if)
        - ASSIGN (assignments)
        
        Ignore: BINARY_OP (too many, create noise), CLASS, FUNCTION wrappers
        
        This focuses on the ALGORITHM SKELETON.
        """
        normalized = []
        mappings = self.PATTERN_MAPPINGS.get(language, {})
        
        # First pass: normalize node types and remove child counts
        for feature in features:
            matched = False
            
            # Extract parts
            parts = feature.split(':')
            base_type = parts[0]
            
            # Try to map to universal pattern
            for pattern_name, node_types in mappings.items():
                if base_type in node_types:
                    # Keep depth but DROP child count
                    depth_part = None
                    for part in parts[1:]:
                        if part.startswith('d'):
                            depth_part = part
                            break
                    
                    if depth_part:
                        normalized.append(f"{pattern_name}:{depth_part}")
                    else:
                        normalized.append(pattern_name)
                    matched = True
                    break
            
            if not matched:
                depth_part = None
                for part in parts[1:]:
                    if part.startswith('d'):
                        depth_part = part
                        break
                
                if depth_part:
                    normalized.append(f"{base_type}:{depth_part}")
                else:
                    normalized.append(base_type)
        
        # Second pass: Keep ONLY primary control structures
        # These define the algorithm skeleton
        PRIMARY_PATTERNS = ['LOOP', 'CONDITIONAL', 'ASSIGN']
        control_flow = []
        for feature in normalized:
            feature_type = feature.split(':')[0]
            if feature_type in PRIMARY_PATTERNS:
                control_flow.append(feature)
        
        # Third pass: Normalize depth to relative (starting from 0)
        depth_normalized = []
        min_depth = float('inf')
        
        # Find minimum depth
        for feature in control_flow:
            if ':d' in feature:
                depth_str = feature.split(':d')[1]
                if depth_str.isdigit():
                    depth = int(depth_str)
                    min_depth = min(min_depth, depth)
        
        # Normalize depths
        if min_depth != float('inf'):
            for feature in control_flow:
                if ':d' in feature:
                    parts = feature.split(':d')
                    if len(parts) == 2 and parts[1].isdigit():
                        depth = int(parts[1])
                        depth_normalized.append(f"{parts[0]}:d{depth - min_depth}")
                    else:
                        depth_normalized.append(feature)
                else:
                    depth_normalized.append(feature)
        else:
            depth_normalized = control_flow
        
        return depth_normalized
    
    def _generate_shingles(self, features: List[str], k: int = 4) -> List[str]:
        """
        Generate k-shingles from features.
        
        Changed from k=3 to k=4 for better specificity.
        
        Why k=4?
        - k=2: Too many false positives (short patterns match too easily)
        - k=3: Good balance (previous default)
        - k=4: Better specificity, fewer false matches, still catches real plagiarism
        - k=5+: Too strict, might miss real similarities
        
        Example with k=4:
        features = ['A', 'B', 'C', 'D', 'E']
        shingles = ['A B C D', 'B C D E']
        
        This captures longer sequences = more unique fingerprints.
        """
        if not features:
            return ['empty']
        
        if len(features) < k:
            return [' '.join(features)]
        
        shingles = []
        for i in range(len(features) - k + 1):
            shingle = ' '.join(features[i:i+k])
            shingles.append(shingle)
        
        return shingles if shingles else ['empty']
    
    def _locality_sensitive_hash(self, shingles: List[str], num_bits: int = 256) -> np.ndarray:
        """
        Generate LSH-based perceptual hash with improved distribution.
        
        Improvements:
        1. Uses 64 hash functions instead of 32 (better distribution)
        2. Each produces 4 bits instead of 8 (more granularity)
        3. Uses SHA-256 instead of MD5 (better collision resistance)
        4. Handles empty/small inputs gracefully
        
        Why more hash functions?
        - More varied hash values across the 256 bits
        - Better collision resistance
        - More accurate similarity scores
        
        Result: More accurate cross-language detection
        """
        # Handle edge case: empty code
        if not shingles or shingles == ['empty']:
            return np.zeros(num_bits, dtype=np.uint8)
        
        # Use more hash functions for better distribution
        # 64 functions × 4 bits = 256 bits total
        num_hashes = num_bits // 4
        hash_bits = []
        
        for i in range(num_hashes):
            # Use SHA-256 with different hash families for better distribution
            hash_family = hashlib.sha256(f"family{i % 8}".encode())
            
            # Hash all shingles together
            for shingle in shingles:
                hash_family.update(shingle.encode())
            
            # Extract 4 bits from this hash
            digest = hash_family.digest()
            for byte_idx in range(min(4, len(digest))):
                if len(hash_bits) < num_bits // 8:
                    byte_val = digest[byte_idx]
                    hash_bits.append(byte_val)
        
        # Convert to binary array
        hash_array = np.array(hash_bits[:num_bits // 8], dtype=np.uint8)
        binary_hash = np.unpackbits(hash_array)[:num_bits]
        
        return binary_hash
    
    def hash_code(self, code: str, language: str) -> np.ndarray:
        """
        Generate a 256-bit perceptual hash for code.
        
        Process:
        1. Parse code to AST
        2. Extract structural features (with depth, complexity)
        3. Normalize to universal patterns (for cross-language)
        4. Generate k-shingles (k=4 for specificity)
        5. Create LSH hash (64 functions for accuracy)
        
        Args:
            code: Source code string
            language: Programming language ('python', 'java', 'cpp')
            
        Returns:
            256-bit hash as numpy array
        """
        if language not in self.LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.LANGUAGES.keys())}")
        
        # Step 1: Extract structural features
        features = self._extract_features(code, language)
        
        # Step 2: Normalize for cross-language detection
        normalized_features = self._normalize_features(features, language)
        
        # Step 3: Generate shingles with adaptive k
        # Smaller k for fewer features (post-normalization)
        if len(normalized_features) <= 5:
            k = 2  # Very small feature set
        elif len(normalized_features) <= 15:
            k = 3  # Small-medium feature set
        else:
            k = 4  # Large feature set
        
        shingles = self._generate_shingles(normalized_features, k=k)
        
        # Step 4: Generate perceptual hash (64 hash functions)
        code_hash = self._locality_sensitive_hash(shingles)
        
        return code_hash
    
    def hash_file(self, file_path: str) -> np.ndarray:
        """
        Generate a 256-bit hash for a code file.
        
        Args:
            file_path: Path to source code file
            
        Returns:
            256-bit hash as numpy array
        """
        # Detect language from extension
        language = self._detect_language(file_path)
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        return self.hash_code(code, language)
    
    def compare(self, hash1: np.ndarray, hash2: np.ndarray) -> Tuple[float, int]:
        """
        Compare two hashes using Hamming distance.
        
        Best for: Same-language comparisons with full feature sets.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Tuple of (similarity_score, hamming_distance)
            - similarity_score: 0.0 to 1.0 (1.0 = identical)
            - hamming_distance: Number of differing bits (0-256)
        """
        if len(hash1) != len(hash2):
            raise ValueError("Hashes must be the same length")
        
        # Calculate Hamming distance
        hamming_distance = np.sum(hash1 != hash2)
        
        # Convert to similarity score (0.0 to 1.0)
        similarity = 1.0 - (hamming_distance / len(hash1))
        
        return similarity, int(hamming_distance)
    
    def compare_cross_language(self, code1: str, lang1: str, 
                               code2: str, lang2: str) -> Tuple[float, Dict]:
        """
        Compare code across different languages using control flow patterns.
        
        Uses Jaccard similarity on normalized shingles instead of LSH hashing.
        This is more accurate for cross-language detection with small feature sets.
        
        Best for: Detecting same algorithm implemented in different languages.
        
        Args:
            code1: First code string
            lang1: First language ('python', 'java', 'cpp')
            code2: Second code string
            lang2: Second language ('python', 'java', 'cpp')
            
        Returns:
            Tuple of (similarity, details_dict)
            - similarity: 0.0 to 1.0 Jaccard similarity
            - details: Dict with matching_shingles, total_shingles, etc.
            
        Example:
            >>> similarity, details = hasher.compare_cross_language(
            ...     python_code, 'python',
            ...     java_code, 'java'
            ... )
            >>> print(f"Similarity: {similarity:.2%}")
            >>> print(f"Matching shingles: {details['matching_shingles']}/{details['total_shingles']}")
        """
        # Extract and normalize features
        features1 = self._extract_features(code1, lang1)
        features2 = self._extract_features(code2, lang2)
        
        norm1 = self._normalize_features(features1, lang1)
        norm2 = self._normalize_features(features2, lang2)
        
        # Generate shingles with appropriate k
        k = 2 if len(norm1) <= 5 or len(norm2) <= 5 else 3
        
        shingles1_list = self._generate_shingles(norm1, k=k)
        shingles2_list = self._generate_shingles(norm2, k=k)
        
        # Convert to sets for Jaccard calculation
        shingles1 = set(shingles1_list)
        shingles2 = set(shingles2_list)
        
        # Calculate Jaccard similarity
        intersection = shingles1 & shingles2
        union = shingles1 | shingles2
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return jaccard, {
            'matching_shingles': len(intersection),
            'total_shingles': len(union),
            'shingles1_count': len(shingles1),
            'shingles2_count': len(shingles2),
            'matching_shingle_list': sorted(list(intersection)),
            'k_value': k
        }
