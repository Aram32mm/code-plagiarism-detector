"""Tests for CodeHasher class."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from code_plagiarism_detector import CodeHasher


class TestCodeHasher:
    """Test suite for CodeHasher."""
    
    @pytest.fixture
    def hasher(self):
        """Create a CodeHasher instance."""
        return CodeHasher()
    
    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
    
    @pytest.fixture
    def similar_python_code(self):
        """Similar Python code with variable name changes."""
        return """
def bubble_sort(items):
    length = len(items)
    for x in range(length):
        for y in range(0, length - x - 1):
            if items[y] > items[y + 1]:
                items[y], items[y + 1] = items[y + 1], items[y]
    return items
"""
    
    @pytest.fixture
    def different_python_code(self):
        """Different Python algorithm."""
        return """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""
    
    def test_hash_code_python(self, hasher, sample_python_code):
        """Test hashing Python code."""
        hash_result = hasher.hash_code(sample_python_code, 'python')
        
        # Verify hash is 256 bits
        assert len(hash_result) == 256
        assert isinstance(hash_result, np.ndarray)
        assert hash_result.dtype == np.uint8
        
        # Verify hash contains binary values
        assert np.all((hash_result == 0) | (hash_result == 1))
    
    def test_hash_code_consistency(self, hasher, sample_python_code):
        """Test that same code produces same hash."""
        hash1 = hasher.hash_code(sample_python_code, 'python')
        hash2 = hasher.hash_code(sample_python_code, 'python')
        
        assert np.array_equal(hash1, hash2)
    
    def test_similar_code_detection(self, hasher, sample_python_code, similar_python_code):
        """Test detection of similar code with variable name changes."""
        hash1 = hasher.hash_code(sample_python_code, 'python')
        hash2 = hasher.hash_code(similar_python_code, 'python')
        
        similarity, hamming_dist = hasher.compare(hash1, hash2)
        
        # Similar code should have high similarity
        assert similarity > 0.7, f"Expected similarity > 0.7, got {similarity}"
        assert hamming_dist < 80, f"Expected hamming distance < 80, got {hamming_dist}"
    
    def test_different_code_detection(self, hasher, sample_python_code, different_python_code):
        """Test detection of different algorithms."""
        hash1 = hasher.hash_code(sample_python_code, 'python')
        hash2 = hasher.hash_code(different_python_code, 'python')
        
        similarity, hamming_dist = hasher.compare(hash1, hash2)
        
        # Different code should have lower similarity
        assert similarity < 0.8, f"Expected similarity < 0.8, got {similarity}"
    
    def test_hash_file(self, hasher, sample_python_code):
        """Test hashing a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_code)
            temp_path = f.name
        
        try:
            hash_result = hasher.hash_file(temp_path)
            
            # Verify hash properties
            assert len(hash_result) == 256
            assert isinstance(hash_result, np.ndarray)
        finally:
            os.unlink(temp_path)
    
    def test_language_detection(self, hasher):
        """Test automatic language detection from file extension."""
        test_cases = [
            ('test.py', 'python'),
            ('test.java', 'java'),
            ('test.cpp', 'cpp'),
            ('test.h', 'cpp'),
        ]
        
        for filename, expected_lang in test_cases:
            detected = hasher._detect_language(filename)
            assert detected == expected_lang
    
    def test_unsupported_language_error(self, hasher):
        """Test error handling for unsupported languages."""
        with pytest.raises(ValueError, match="Unsupported language"):
            hasher.hash_code("code", 'unsupported')
    
    def test_unsupported_extension_error(self, hasher):
        """Test error handling for unsupported file extensions."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            hasher._detect_language("test.xyz")
    
    def test_compare_different_length_hashes(self, hasher):
        """Test error when comparing hashes of different lengths."""
        hash1 = np.array([0, 1, 0, 1])
        hash2 = np.array([1, 0, 1])
        
        with pytest.raises(ValueError, match="Hashes must be the same length"):
            hasher.compare(hash1, hash2)
    
    def test_identical_code_similarity(self, hasher, sample_python_code):
        """Test that identical code has 100% similarity."""
        hash1 = hasher.hash_code(sample_python_code, 'python')
        hash2 = hasher.hash_code(sample_python_code, 'python')
        
        similarity, hamming_dist = hasher.compare(hash1, hash2)
        
        assert similarity == 1.0
        assert hamming_dist == 0
    
    def test_java_code_hashing(self, hasher):
        """Test hashing Java code."""
        java_code = """
public class BubbleSort {
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}
"""
        hash_result = hasher.hash_code(java_code, 'java')
        assert len(hash_result) == 256
    
    def test_cpp_code_hashing(self, hasher):
        """Test hashing C++ code."""
        cpp_code = """
#include <iostream>
#include <vector>

void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}
"""
        hash_result = hasher.hash_code(cpp_code, 'cpp')
        assert len(hash_result) == 256


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
