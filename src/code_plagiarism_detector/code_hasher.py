"""CodeHasher class for generating perceptual hashes of code using AST."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tree_sitter import Language, Parser


class CodeHasher:
    """Generate and compare perceptual hashes of code based on AST structure."""

    # Language extensions mapping
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp',
    }

    def __init__(self):
        """Initialize CodeHasher with tree-sitter parsers."""
        self.parsers = {}
        self.languages = {}
        self._setup_parsers()

    def _setup_parsers(self):
        """Set up tree-sitter parsers for supported languages."""
        try:
            import tree_sitter_python
            import tree_sitter_java
            import tree_sitter_cpp

            # Set up Python parser
            self.languages['python'] = Language(tree_sitter_python.language())
            python_parser = Parser(self.languages['python'])
            self.parsers['python'] = python_parser

            # Set up Java parser
            self.languages['java'] = Language(tree_sitter_java.language())
            java_parser = Parser(self.languages['java'])
            self.parsers['java'] = java_parser

            # Set up C++ parser
            self.languages['cpp'] = Language(tree_sitter_cpp.language())
            cpp_parser = Parser(self.languages['cpp'])
            self.parsers['cpp'] = cpp_parser

        except ImportError as e:
            raise ImportError(
                f"Failed to import tree-sitter language bindings: {e}. "
                "Please install tree-sitter-python, tree-sitter-java, and tree-sitter-cpp."
            )

    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(ext)

    def _extract_features(self, tree, language: str) -> Dict[str, List[str]]:
        """Extract structural features from AST.
        
        Args:
            tree: Tree-sitter parse tree
            language: Programming language name
            
        Returns:
            Dictionary with extracted features (functions, control_flow, imports)
        """
        features = {
            'functions': [],
            'control_flow': [],
            'imports': [],
            'structure': []
        }

        def traverse(node):
            """Recursively traverse AST and extract features."""
            node_type = node.type

            # Extract function/method definitions
            if node_type in ['function_definition', 'method_declaration', 
                            'function_declaration', 'constructor_declaration']:
                features['functions'].append(node_type)

            # Extract control flow structures
            elif node_type in ['if_statement', 'for_statement', 'while_statement',
                              'switch_statement', 'try_statement', 'catch_clause',
                              'with_statement', 'match_statement']:
                features['control_flow'].append(node_type)

            # Extract imports
            elif node_type in ['import_statement', 'import_from_statement',
                              'import_declaration', 'using_declaration']:
                features['imports'].append(node_type)

            # Extract general structure
            features['structure'].append(node_type)

            # Recursively process children
            for child in node.children:
                traverse(child)

        # Start traversal from root
        traverse(tree.root_node)
        return features

    def _generate_hash(self, features: Dict[str, List[str]]) -> np.ndarray:
        """Generate 256-bit perceptual hash using LSH (Locality Sensitive Hashing).
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            256-bit hash as numpy array
        """
        # Create feature vector from structural elements
        feature_strings = []
        
        # Concatenate all feature types
        for feature_type in ['functions', 'control_flow', 'imports', 'structure']:
            feature_strings.extend(features.get(feature_type, []))

        # Convert to string and hash
        feature_str = ''.join(feature_strings)
        
        # Use SHA-256 for base hash
        hash_bytes = hashlib.sha256(feature_str.encode()).digest()
        
        # Convert to bit array (256 bits)
        hash_bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
        
        # Apply simhash-style LSH for better similarity detection
        # Use feature counts to create weighted hash
        feature_weights = np.zeros(256, dtype=np.float64)
        
        for feature in feature_strings:
            # Hash each feature individually
            feature_hash = hashlib.sha256(feature.encode()).digest()
            feature_bits = np.unpackbits(np.frombuffer(feature_hash, dtype=np.uint8))
            
            # Add or subtract based on bit value
            feature_weights += np.where(feature_bits == 1, 1, -1)
        
        # Convert weights to final hash bits
        final_hash = (feature_weights >= 0).astype(np.uint8)
        
        return final_hash

    def hash_code(self, code: str, language: str) -> np.ndarray:
        """Generate hash for code string.
        
        Args:
            code: Source code string
            language: Programming language ('python', 'java', 'cpp')
            
        Returns:
            256-bit hash as numpy array
            
        Raises:
            ValueError: If language is not supported
        """
        if language not in self.parsers:
            raise ValueError(f"Unsupported language: {language}. "
                           f"Supported: {list(self.parsers.keys())}")

        # Parse code
        parser = self.parsers[language]
        tree = parser.parse(code.encode())

        # Extract features
        features = self._extract_features(tree, language)

        # Generate hash
        return self._generate_hash(features)

    def hash_file(self, file_path: str) -> np.ndarray:
        """Generate hash for code file.
        
        Args:
            file_path: Path to source code file
            
        Returns:
            256-bit hash as numpy array
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If language cannot be detected or is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect language
        language = self._detect_language(file_path)
        if language is None:
            raise ValueError(f"Could not detect language for file: {file_path}")

        # Read file
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Generate hash
        return self.hash_code(code, language)

    def compare(self, hash1: np.ndarray, hash2: np.ndarray) -> Tuple[float, int]:
        """Compare two hashes using Hamming distance.
        
        Args:
            hash1: First hash (256-bit numpy array)
            hash2: Second hash (256-bit numpy array)
            
        Returns:
            Tuple of (similarity_score, hamming_distance)
            - similarity_score: 0.0 to 1.0, where 1.0 is identical
            - hamming_distance: Number of differing bits (0-256)
            
        Raises:
            ValueError: If hashes have different lengths
        """
        if len(hash1) != len(hash2):
            raise ValueError(f"Hash lengths must match: {len(hash1)} vs {len(hash2)}")

        # Calculate Hamming distance (number of differing bits)
        hamming_dist = int(np.sum(hash1 != hash2))

        # Calculate similarity score (0.0 to 1.0)
        similarity = 1.0 - (hamming_dist / len(hash1))

        return similarity, hamming_dist
