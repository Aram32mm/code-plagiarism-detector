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
            'with_statement', 'import_statement', 'import_from_statement'
        ],
        'java': [
            'method_declaration', 'class_declaration', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'enhanced_for_statement', 'import_declaration'
        ],
        'cpp': [
            'function_definition', 'class_specifier', 'if_statement',
            'for_statement', 'while_statement', 'try_statement',
            'using_declaration', 'preproc_include'
        ]
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
    
    def _extract_features(self, code: str, language: str) -> List[str]:
        """Extract structural features from AST."""
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, 'utf-8'))
        
        features = []
        structural_nodes = self.STRUCTURAL_NODES[language]
        
        def traverse(node):
            """Recursively traverse AST and collect structural features."""
            if node.type in structural_nodes:
                # Add node type and depth as feature
                features.append(f"{node.type}@{node.start_point[0]}")
            
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return features
    
    def _generate_shingles(self, features: List[str], k: int = 3) -> List[str]:
        """Generate k-shingles from features."""
        if len(features) < k:
            return [' '.join(features)]
        return [' '.join(features[i:i+k]) for i in range(len(features) - k + 1)]
    
    def _locality_sensitive_hash(self, shingles: List[str], num_bits: int = 256) -> np.ndarray:
        """Generate LSH-based perceptual hash."""
        # Use multiple hash functions for LSH
        num_hashes = num_bits // 8  # 32 hash functions for 256 bits
        hash_values = []
        
        for i in range(num_hashes):
            # Create hash with different seeds
            h = hashlib.md5(f"seed{i}".encode())
            for shingle in shingles:
                h.update(shingle.encode())
            # Get 8 bits from each hash
            hash_bytes = h.digest()
            hash_values.append(hash_bytes[0])
        
        # Convert to binary array
        binary_hash = np.unpackbits(np.array(hash_values, dtype=np.uint8))
        return binary_hash
    
    def hash_code(self, code: str, language: str) -> np.ndarray:
        """
        Generate a 256-bit perceptual hash for code.
        
        Args:
            code: Source code string
            language: Programming language ('python', 'java', 'cpp')
            
        Returns:
            256-bit hash as numpy array
        """
        if language not in self.LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.LANGUAGES.keys())}")
        
        # Extract structural features
        features = self._extract_features(code, language)
        
        # Generate shingles
        shingles = self._generate_shingles(features)
        
        # Generate perceptual hash
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
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Tuple of (similarity_score, hamming_distance)
        """
        if len(hash1) != len(hash2):
            raise ValueError("Hashes must be the same length")
        
        # Calculate Hamming distance
        hamming_distance = np.sum(hash1 != hash2)
        
        # Convert to similarity score (0.0 to 1.0)
        similarity = 1.0 - (hamming_distance / len(hash1))
        
        return similarity, int(hamming_distance)
