"""Code plagiarism detector using AST fingerprinting and perceptual hashing."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp


@dataclass
class ComparisonResult:
    """Complete plagiarism detection result."""
    
    similarity: float
    plagiarism_detected: bool
    confidence: str
    syntactic_similarity: float
    hamming_distance: int
    structural_similarity: float
    matching_patterns: List[str]
    pattern_match_ratio: str
    lang1: str
    lang2: str
    method_used: str
    
    def __repr__(self):
        return (
            f"ComparisonResult(similarity={self.similarity:.1%}, "
            f"plagiarism={self.plagiarism_detected}, "
            f"confidence='{self.confidence}')"
        )


class CodeHasher:
    """Generate and compare perceptual hashes of code based on AST structure."""
    
    LANGUAGES = {
        'python': {'parser': tspython, 'extensions': ['.py']},
        'java': {'parser': tsjava, 'extensions': ['.java']},
        'cpp': {'parser': tscpp, 'extensions': ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']}
    }
    
    STRUCTURAL_NODES = {
        'python': [
            'function_definition', 'class_definition', 'if_statement', 
            'for_statement', 'while_statement', 'try_statement',
            'with_statement', 'assignment', 'comparison_operator', 'binary_operator'
        ],
        'java': [
            'method_declaration', 'class_declaration', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'enhanced_for_statement', 'assignment_expression', 'binary_expression'
        ],
        'cpp': [
            'function_definition', 'class_specifier', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'assignment_expression', 'binary_expression'
        ]
    }
    
    # Control flow patterns for cross-language matching
    CONTROL_FLOW_NODES = {
        'python': {
            'LOOP': ['for_statement', 'while_statement'],
            'COND': ['if_statement'],
        },
        'java': {
            'LOOP': ['for_statement', 'while_statement', 'enhanced_for_statement', 'do_statement'],
            'COND': ['if_statement'],
        },
        'cpp': {
            'LOOP': ['for_statement', 'while_statement', 'for_range_loop', 'do_statement'],
            'COND': ['if_statement'],
        }
    }
    
    def __init__(self):
        """Initialize parsers for each language."""
        self.parsers = {}
        for lang_name, lang_config in self.LANGUAGES.items():
            parser = Parser(Language(lang_config['parser'].language()))
            self.parsers[lang_name] = parser
    
    def compare(self, code1: str, lang1: str, code2: str, lang2: str) -> ComparisonResult:
        """Compare two code snippets and return comprehensive results."""
        if lang1 not in self.LANGUAGES:
            raise ValueError(f"Unsupported language: {lang1}")
        if lang2 not in self.LANGUAGES:
            raise ValueError(f"Unsupported language: {lang2}")
        
        # Syntactic analysis (hash-based)
        hash1 = self.hash_code(code1, lang1)
        hash2 = self.hash_code(code2, lang2)
        hamming_dist = int(np.sum(hash1 != hash2))
        syntactic_sim = 1.0 - (hamming_dist / len(hash1))
        
        # Structural analysis (pattern-based)
        structural_sim, patterns, match_ratio = self._compare_structure(
            code1, lang1, code2, lang2
        )
        
        # Use best result
        best_sim = max(syntactic_sim, structural_sim)
        method_used = 'syntactic' if syntactic_sim >= structural_sim else 'structural'
        
        # Confidence thresholds
        if structural_sim >= 0.60 or best_sim >= 0.85:
            confidence = 'high'
        elif structural_sim >= 0.40 or best_sim >= 0.70:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return ComparisonResult(
            similarity=best_sim,
            plagiarism_detected=best_sim > 0.50 or structural_sim > 0.50,
            confidence=confidence,
            syntactic_similarity=syntactic_sim,
            hamming_distance=hamming_dist,
            structural_similarity=structural_sim,
            matching_patterns=patterns,
            pattern_match_ratio=match_ratio,
            lang1=lang1,
            lang2=lang2,
            method_used=method_used
        )
    
    def compare_files(self, file1: str, file2: str) -> ComparisonResult:
        """Compare two code files."""
        lang1 = self._detect_language(file1)
        lang2 = self._detect_language(file2)
        
        with open(file1, 'r', encoding='utf-8') as f:
            code1 = f.read()
        with open(file2, 'r', encoding='utf-8') as f:
            code2 = f.read()
        
        return self.compare(code1, lang1, code2, lang2)
    
    def hash_code(self, code: str, language: str) -> np.ndarray:
        """Generate a 256-bit perceptual hash for code."""
        if language not in self.LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.LANGUAGES.keys())}")
        
        features = self._extract_features(code, language)
        normalized = self._normalize_features(features, language)
        
        k = 2 if len(normalized) <= 5 else 3 if len(normalized) <= 15 else 4
        shingles = self._generate_shingles(normalized, k)
        
        return self._locality_sensitive_hash(shingles)
    
    def hash_file(self, file_path: str) -> np.ndarray:
        """Generate a 256-bit hash for a code file."""
        language = self._detect_language(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return self.hash_code(code, language)
    
    def compare_hashes(self, hash1: np.ndarray, hash2: np.ndarray) -> Tuple[float, int]:
        """Compare two hashes using Hamming distance."""
        if len(hash1) != len(hash2):
            raise ValueError("Hashes must be the same length")
        
        hamming_distance = int(np.sum(hash1 != hash2))
        similarity = 1.0 - (hamming_distance / len(hash1))
        
        return similarity, hamming_distance
    
    def _compare_structure(self, code1: str, lang1: str, 
                           code2: str, lang2: str) -> Tuple[float, List[str], str]:
        """Compare structural patterns - ONLY LOOP and COND, normalized depths."""
        patterns1 = self._extract_control_flow(code1, lang1)
        patterns2 = self._extract_control_flow(code2, lang2)
        
        if not patterns1 and not patterns2:
            return 1.0, [], "0/0"
        if not patterns1 or not patterns2:
            return 0.0, [], "0/0"
        
        # Generate shingles
        shingles1 = set(self._generate_shingles(patterns1, k=2))
        shingles2 = set(self._generate_shingles(patterns2, k=2))
        
        shingles1.discard('empty')
        shingles2.discard('empty')
        
        if not shingles1 or not shingles2:
            # Fallback: direct pattern comparison
            set1 = set(patterns1)
            set2 = set(patterns2)
            intersection = set1 & set2
            union = set1 | set2
            if union:
                return len(intersection) / len(union), sorted(list(intersection)), f"{len(intersection)}/{len(union)}"
            return 0.0, [], "0/0"
        
        intersection = shingles1 & shingles2
        union = shingles1 | shingles2
        
        similarity = len(intersection) / len(union) if union else 0.0
        patterns = sorted(list(intersection))
        ratio = f"{len(intersection)}/{len(union)}"
        
        return similarity, patterns, ratio
    
    def _extract_control_flow(self, code: str, language: str) -> List[str]:
        """
        Extract ONLY loop and conditional patterns, with normalized depths.
        
        This strips CLASS/FUNC wrappers and focuses on algorithm structure:
        - LOOP (for, while)
        - COND (if)
        
        Depths are relative to the first control flow node found.
        """
        if not code.strip():
            return []
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, 'utf-8'))
        mappings = self.CONTROL_FLOW_NODES.get(language, {})
        
        # Build reverse mapping
        node_to_pattern = {}
        for pattern_name, node_types in mappings.items():
            for node_type in node_types:
                node_to_pattern[node_type] = pattern_name
        
        patterns = []
        
        def traverse(node, depth=0):
            # Only capture LOOP and COND, ignore CLASS/FUNC
            if node.type in node_to_pattern:
                patterns.append((node_to_pattern[node.type], depth))
            
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(tree.root_node)
        
        if not patterns:
            return []
        
        # Normalize depths relative to FIRST control flow pattern
        min_depth = min(p[1] for p in patterns)
        normalized = [f"{p[0]}:d{p[1] - min_depth}" for p in patterns]
        
        return normalized
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        for lang, config in self.LANGUAGES.items():
            if ext in config['extensions']:
                return lang
        raise ValueError(f"Unsupported file extension: {ext}")
    
    def _extract_features(self, code: str, language: str) -> List[str]:
        """Extract structural features from AST."""
        if not code.strip():
            return []
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, 'utf-8'))
        
        features = []
        structural_nodes = self.STRUCTURAL_NODES[language]
        
        def traverse(node, depth=0):
            if node.type in structural_nodes:
                features.append(f"{node.type}:d{depth}")
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(tree.root_node)
        return features
    
    def _normalize_features(self, features: List[str], language: str) -> List[str]:
        """Normalize to universal patterns."""
        if not features:
            return []
        
        mappings = self.CONTROL_FLOW_NODES.get(language, {})
        
        node_to_pattern = {}
        for pattern_name, node_types in mappings.items():
            for node_type in node_types:
                node_to_pattern[node_type] = pattern_name
        
        normalized = []
        for feature in features:
            parts = feature.split(':')
            base_type = parts[0]
            depth_part = parts[1] if len(parts) > 1 else None
            
            if base_type in node_to_pattern:
                pattern = node_to_pattern[base_type]
                if depth_part:
                    normalized.append(f"{pattern}:{depth_part}")
                else:
                    normalized.append(pattern)
            elif depth_part:
                normalized.append(f"{base_type}:{depth_part}")
        
        if not normalized:
            return []
        
        # Normalize depths
        min_depth = float('inf')
        for f in normalized:
            if ':d' in f:
                try:
                    d = int(f.split(':d')[1])
                    min_depth = min(min_depth, d)
                except (ValueError, IndexError):
                    pass
        
        if min_depth == float('inf'):
            return normalized
        
        result = []
        for f in normalized:
            if ':d' in f:
                try:
                    parts = f.split(':d')
                    d = int(parts[1])
                    result.append(f"{parts[0]}:d{d - min_depth}")
                except (ValueError, IndexError):
                    result.append(f)
            else:
                result.append(f)
        
        return result
    
    def _generate_shingles(self, features: List[str], k: int = 3) -> List[str]:
        """Generate k-shingles from features."""
        if not features:
            return ['empty']
        if len(features) < k:
            return [' '.join(features)] if features else ['empty']
        return [' '.join(features[i:i+k]) for i in range(len(features) - k + 1)]
    
    def _locality_sensitive_hash(self, shingles: List[str], num_bits: int = 256) -> np.ndarray:
        """Generate LSH-based 256-bit hash."""
        if not shingles or shingles == ['empty']:
            return np.zeros(num_bits, dtype=np.uint8)
        
        hash_bytes = []
        for i in range(num_bits // 8):
            h = hashlib.sha256(f"salt{i}".encode())
            for s in shingles:
                h.update(s.encode())
            hash_bytes.append(h.digest()[0])
        
        return np.unpackbits(np.array(hash_bytes, dtype=np.uint8))
    
    def debug_patterns(self, code: str, language: str) -> dict:
        """Debug helper to see extracted patterns."""
        return {
            'control_flow': self._extract_control_flow(code, language),
        }
