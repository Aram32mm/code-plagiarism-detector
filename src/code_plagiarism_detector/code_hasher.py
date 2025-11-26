"""Code plagiarism detector using AST fingerprinting and perceptual hashing."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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
    debug_info: Dict = field(default_factory=dict)
    
    def __repr__(self):
        return (
            f"ComparisonResult(similarity={self.similarity:.1%}, "
            f"plagiarism={self.plagiarism_detected}, "
            f"confidence='{self.confidence}')"
        )


@dataclass
class LanguageConfig:
    """Configuration for a programming language."""
    parser_module: object
    extensions: List[str]
    structural_nodes: List[str]
    pattern_mappings: Dict[str, List[str]]


@dataclass
class DetectorConfig:
    """Configuration for detection thresholds and behavior."""
    high_confidence_threshold: float = 0.70      # Stricter
    medium_confidence_threshold: float = 0.50    # Stricter
    plagiarism_threshold: float = 0.60           # Stricter
    hash_bits: int = 256
    shingle_sizes: Tuple[int, int, int] = (2, 3, 4)
    shingle_thresholds: Tuple[int, int] = (5, 15)


# Default language configurations
DEFAULT_LANGUAGES: Dict[str, LanguageConfig] = {
    'python': LanguageConfig(
        parser_module=tspython,
        extensions=['.py'],
        structural_nodes=[
            'function_definition', 'class_definition', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'with_statement', 'return_statement', 'call'
        ],
        pattern_mappings={
            'LOOP': ['for_statement', 'while_statement'],
            'COND': ['if_statement'],
        }
    ),
    'java': LanguageConfig(
        parser_module=tsjava,
        extensions=['.java'],
        structural_nodes=[
            'method_declaration', 'class_declaration', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'enhanced_for_statement', 'return_statement', 'method_invocation'
        ],
        pattern_mappings={
            'LOOP': ['for_statement', 'while_statement', 'enhanced_for_statement', 'do_statement'],
            'COND': ['if_statement'],
        }
    ),
    'cpp': LanguageConfig(
        parser_module=tscpp,
        extensions=['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp'],
        structural_nodes=[
            'function_definition', 'class_specifier', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'return_statement', 'call_expression'
        ],
        pattern_mappings={
            'LOOP': ['for_statement', 'while_statement', 'for_range_loop', 'do_statement'],
            'COND': ['if_statement'],
        }
    ),
}


class CodeHasher:
    """Generate and compare perceptual hashes of code based on AST structure."""
    
    def __init__(self,
                 languages: Optional[Dict[str, LanguageConfig]] = None,
                 config: Optional[DetectorConfig] = None):
        """
        Initialize with optional custom configurations.
        
        Args:
            languages: Custom language configurations (defaults to Python/Java/C++)
            config: Detection thresholds and parameters
        """
        self.languages = languages or DEFAULT_LANGUAGES.copy()
        self.config = config or DetectorConfig()
        
        self.parsers: Dict[str, Parser] = {}
        self._node_to_pattern: Dict[str, Dict[str, str]] = {}
        
        self._init_parsers()
        self._build_mappings()
    
    def _init_parsers(self):
        """Initialize parsers for all configured languages."""
        for lang_name, lang_config in self.languages.items():
            parser = Parser(Language(lang_config.parser_module.language()))
            self.parsers[lang_name] = parser
    
    def _build_mappings(self):
        """Build reverse node-to-pattern mappings for fast lookup."""
        self._node_to_pattern = {}
        
        for lang_name, lang_config in self.languages.items():
            self._node_to_pattern[lang_name] = {}
            for pattern, nodes in lang_config.pattern_mappings.items():
                for node in nodes:
                    self._node_to_pattern[lang_name][node] = pattern
    
    def register_language(self, name: str, config: LanguageConfig):
        """
        Register a new language at runtime.
        
        Args:
            name: Language identifier
            config: Language configuration
        """
        self.languages[name] = config
        parser = Parser(Language(config.parser_module.language()))
        self.parsers[name] = parser
        self._build_mappings()
    
    def update_patterns(self, language: str, patterns: Dict[str, List[str]]):
        """
        Update pattern mappings for a language.
        
        Args:
            language: Language to update
            patterns: New patterns to add/update
        """
        if language not in self.languages:
            raise ValueError(f"Unknown language: {language}")
        self.languages[language].pattern_mappings.update(patterns)
        self._build_mappings()
    
    def compare(self, code1: str, lang1: str, code2: str, lang2: str) -> ComparisonResult:
        """
        Compare two code snippets and return comprehensive results.
        
        Args:
            code1: First code string
            lang1: Language of first code
            code2: Second code string
            lang2: Language of second code
            
        Returns:
            ComparisonResult with all detection metrics
        """
        if lang1 not in self.languages:
            raise ValueError(f"Unsupported language: {lang1}. Available: {list(self.languages.keys())}")
        if lang2 not in self.languages:
            raise ValueError(f"Unsupported language: {lang2}. Available: {list(self.languages.keys())}")
        
        # Syntactic analysis (hash-based)
        hash1 = self.hash_code(code1, lang1)
        hash2 = self.hash_code(code2, lang2)
        hamming_dist = int(np.sum(hash1 != hash2))
        syntactic_sim = 1.0 - (hamming_dist / len(hash1))
        
        # Structural analysis (pattern-based)
        structural_sim, patterns, match_ratio, debug = self._compare_structure(
            code1, lang1, code2, lang2
        )
        
        # Best result
        best_sim = max(syntactic_sim, structural_sim)
        method_used = 'syntactic' if syntactic_sim >= structural_sim else 'structural'
        
        # Confidence based on config thresholds - ALIGNED WITH UI
        if structural_sim >= 0.70 or best_sim >= 0.90:
            confidence = 'high'
        elif structural_sim >= 0.50 or best_sim >= 0.75:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        plagiarism_detected = (
            best_sim > 0.60 or 
            structural_sim > 0.60
        )
        
        return ComparisonResult(
            similarity=best_sim,
            plagiarism_detected=plagiarism_detected,
            confidence=confidence,
            syntactic_similarity=syntactic_sim,
            hamming_distance=hamming_dist,
            structural_similarity=structural_sim,
            matching_patterns=patterns,
            pattern_match_ratio=match_ratio,
            lang1=lang1,
            lang2=lang2,
            method_used=method_used,
            debug_info=debug
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
        """
        Generate a perceptual hash for code.
        
        Args:
            code: Source code string
            language: Programming language
            
        Returns:
            Hash as numpy array of bits
        """
        if language not in self.languages:
            raise ValueError(f"Unsupported language: {language}")
        
        features = self._extract_features(code, language)
        normalized = self._normalize_features(features, language)
        
        # Adaptive shingle size
        small_k, medium_k, large_k = self.config.shingle_sizes
        thresh_small, thresh_large = self.config.shingle_thresholds
        
        if len(normalized) <= thresh_small:
            k = small_k
        elif len(normalized) <= thresh_large:
            k = medium_k
        else:
            k = large_k
        
        shingles = self._generate_shingles(normalized, k)
        return self._locality_sensitive_hash(shingles, self.config.hash_bits)
    
    def hash_file(self, file_path: str) -> np.ndarray:
        """Generate hash for a code file."""
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
                           code2: str, lang2: str) -> Tuple[float, List[str], str, Dict]:
        """Compare structural patterns with debug info."""
        patterns1 = self._extract_control_flow(code1, lang1)
        patterns2 = self._extract_control_flow(code2, lang2)
        
        debug = {
            'patterns1': patterns1,
            'patterns2': patterns2,
        }
        
        if not patterns1 and not patterns2:
            return 1.0, [], "0/0", debug
        if not patterns1 or not patterns2:
            return 0.0, [], "0/0", debug
        
        # Generate shingles
        shingles1 = set(self._generate_shingles(patterns1, k=2))
        shingles2 = set(self._generate_shingles(patterns2, k=2))
        
        shingles1.discard('empty')
        shingles2.discard('empty')
        
        debug['shingles1'] = sorted(shingles1)
        debug['shingles2'] = sorted(shingles2)
        
        if not shingles1 or not shingles2:
            set1 = set(patterns1)
            set2 = set(patterns2)
            intersection = set1 & set2
            union = set1 | set2
            if union:
                return len(intersection) / len(union), sorted(list(intersection)), f"{len(intersection)}/{len(union)}", debug
            return 0.0, [], "0/0", debug
        
        intersection = shingles1 & shingles2
        union = shingles1 | shingles2
        
        similarity = len(intersection) / len(union) if union else 0.0
        patterns = sorted(list(intersection))
        ratio = f"{len(intersection)}/{len(union)}"
        
        debug['intersection'] = patterns
        
        return similarity, patterns, ratio, debug
    
    def _extract_control_flow(self, code: str, language: str) -> List[str]:
        """Extract control flow patterns using language config."""
        if not code.strip():
            return []
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, 'utf-8'))
        node_to_pattern = self._node_to_pattern.get(language, {})
        
        patterns = []
        
        def traverse(node, depth=0):
            if node.type in node_to_pattern:
                patterns.append((node_to_pattern[node.type], depth))
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(tree.root_node)
        
        if not patterns:
            return []
        
        min_depth = min(p[1] for p in patterns)
        return [f"{p[0]}:d{p[1] - min_depth}" for p in patterns]
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        for lang, config in self.languages.items():
            if ext in config.extensions:
                return lang
        raise ValueError(f"Unsupported file extension: {ext}")
    
    def _extract_features(self, code: str, language: str) -> List[str]:
        """Extract structural features from AST."""
        if not code.strip():
            return []
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, 'utf-8'))
        
        structural_nodes = self.languages[language].structural_nodes
        features = []
        
        def traverse(node, depth=0):
            if node.type in structural_nodes:
                features.append(f"{node.type}:d{depth}")
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(tree.root_node)
        return features
    
    def _normalize_features(self, features: List[str], language: str) -> List[str]:
        """Normalize features to universal patterns."""
        if not features:
            return []
        
        node_to_pattern = self._node_to_pattern.get(language, {})
        normalized = []
        
        for feature in features:
            parts = feature.split(':')
            base_type = parts[0]
            depth_part = parts[1] if len(parts) > 1 else None
            
            if base_type in node_to_pattern:
                pattern = node_to_pattern[base_type]
                normalized.append(f"{pattern}:{depth_part}" if depth_part else pattern)
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
        """Generate LSH-based hash."""
        if not shingles or shingles == ['empty']:
            return np.zeros(num_bits, dtype=np.uint8)
        
        hash_bytes = []
        for i in range(num_bits // 8):
            h = hashlib.sha256(f"salt{i}".encode())
            for s in shingles:
                h.update(s.encode())
            hash_bytes.append(h.digest()[0])
        
        return np.unpackbits(np.array(hash_bytes, dtype=np.uint8))
    
    def debug_patterns(self, code: str, language: str) -> Dict:
        """Debug helper to see extracted patterns."""
        return {
            'features': self._extract_features(code, language),
            'normalized': self._normalize_features(
                self._extract_features(code, language), language
            ),
            'control_flow': self._extract_control_flow(code, language),
        }
