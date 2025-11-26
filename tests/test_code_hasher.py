"""Tests for CodeHasher class."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from code_plagiarism_detector import CodeHasher, ComparisonResult


# ============================================================================
# Test Fixtures
# ============================================================================

PYTHON_BUBBLE_SORT = """
class Solution:
    def bubble_sort(self, arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
"""

PYTHON_BUBBLE_SORT_RENAMED = """
class Solution:
    def bubble_sort(self, items):
        length = len(items)
        for x in range(length):
            for y in range(0, length - x - 1):
                if items[y] > items[y + 1]:
                    items[y], items[y + 1] = items[y + 1], items[y]
        return items
"""

PYTHON_QUICK_SORT = """
class Solution:
    def quick_sort(self, arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return self.quick_sort(left) + middle + self.quick_sort(right)
"""

JAVA_BUBBLE_SORT = """
public class Solution {
    public void bubbleSort(int[] arr) {
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

CPP_BUBBLE_SORT = """
class Solution {
public:
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
};
"""


# ============================================================================
# Hash Generation Tests
# ============================================================================

class TestHashGeneration:
    """Test hash generation functionality."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_hash_returns_256_bits(self, hasher):
        """Hash should be exactly 256 bits."""
        hash_result = hasher.hash_code(PYTHON_BUBBLE_SORT, 'python')
        assert len(hash_result) == 256
    
    def test_hash_is_numpy_array(self, hasher):
        """Hash should be a numpy array."""
        hash_result = hasher.hash_code(PYTHON_BUBBLE_SORT, 'python')
        assert isinstance(hash_result, np.ndarray)
        assert hash_result.dtype == np.uint8
    
    def test_hash_is_binary(self, hasher):
        """Hash should contain only 0s and 1s."""
        hash_result = hasher.hash_code(PYTHON_BUBBLE_SORT, 'python')
        assert np.all((hash_result == 0) | (hash_result == 1))
    
    def test_hash_is_deterministic(self, hasher):
        """Same code should produce same hash."""
        hash1 = hasher.hash_code(PYTHON_BUBBLE_SORT, 'python')
        hash2 = hasher.hash_code(PYTHON_BUBBLE_SORT, 'python')
        assert np.array_equal(hash1, hash2)
    
    def test_hash_python(self, hasher):
        """Python code hashing works."""
        hash_result = hasher.hash_code(PYTHON_BUBBLE_SORT, 'python')
        assert len(hash_result) == 256
    
    def test_hash_java(self, hasher):
        """Java code hashing works."""
        hash_result = hasher.hash_code(JAVA_BUBBLE_SORT, 'java')
        assert len(hash_result) == 256
    
    def test_hash_cpp(self, hasher):
        """C++ code hashing works."""
        hash_result = hasher.hash_code(CPP_BUBBLE_SORT, 'cpp')
        assert len(hash_result) == 256
    
    def test_hash_empty_code(self, hasher):
        """Empty code should not crash."""
        hash_result = hasher.hash_code("", 'python')
        assert len(hash_result) == 256
    
    def test_hash_minimal_code(self, hasher):
        """Minimal code should work."""
        hash_result = hasher.hash_code("x = 1", 'python')
        assert len(hash_result) == 256


# ============================================================================
# File Handling Tests
# ============================================================================

class TestFileHandling:
    """Test file-based operations."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_hash_file(self, hasher):
        """File hashing works."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(PYTHON_BUBBLE_SORT)
            temp_path = f.name
        
        try:
            hash_result = hasher.hash_file(temp_path)
            assert len(hash_result) == 256
            assert isinstance(hash_result, np.ndarray)
        finally:
            os.unlink(temp_path)
    
    def test_language_detection_python(self, hasher):
        """Detects Python from .py extension."""
        assert hasher._detect_language('test.py') == 'python'
    
    def test_language_detection_java(self, hasher):
        """Detects Java from .java extension."""
        assert hasher._detect_language('test.java') == 'java'
    
    def test_language_detection_cpp(self, hasher):
        """Detects C++ from various extensions."""
        assert hasher._detect_language('test.cpp') == 'cpp'
        assert hasher._detect_language('test.cc') == 'cpp'
        assert hasher._detect_language('test.h') == 'cpp'
        assert hasher._detect_language('test.hpp') == 'cpp'
    
    def test_unsupported_extension_raises(self, hasher):
        """Unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            hasher._detect_language('test.xyz')
    
    def test_unsupported_language_raises(self, hasher):
        """Unsupported language raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported language"):
            hasher.hash_code("code", 'unsupported')


# ============================================================================
# Comparison Tests - Same Language
# ============================================================================

class TestSameLanguageComparison:
    """Test same-language plagiarism detection."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_identical_code_100_percent(self, hasher):
        """Identical code should have 100% similarity."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', PYTHON_BUBBLE_SORT, 'python')
        
        assert result.similarity == 1.0
        assert result.hamming_distance == 0
        assert result.plagiarism_detected is True
        assert result.confidence == 'high'
    
    def test_renamed_variables_high_similarity(self, hasher):
        """Renamed variables should still be detected."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', PYTHON_BUBBLE_SORT_RENAMED, 'python')
        
        assert result.similarity >= 0.85
        assert result.plagiarism_detected is True
        assert result.confidence == 'high'
    
    def test_different_algorithms_lower_similarity(self, hasher):
        """Different algorithms should have lower similarity."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', PYTHON_QUICK_SORT, 'python')
        
        assert result.similarity < 0.85
        assert result.confidence != 'high'


# ============================================================================
# Comparison Tests - Cross Language (All Bubble Sort = High Confidence)
# ============================================================================

class TestCrossLanguageComparison:
    """Test cross-language plagiarism detection. All bubble sort should be HIGH confidence."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_python_vs_java_bubble_sort(self, hasher):
        """Python vs Java bubble sort should be HIGH confidence."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', JAVA_BUBBLE_SORT, 'java')
        
        assert result.plagiarism_detected is True
        assert result.confidence == 'high', \
            f"Expected 'high', got '{result.confidence}' (structural: {result.structural_similarity:.1%})"
        assert len(result.matching_patterns) > 0
    
    def test_python_vs_cpp_bubble_sort(self, hasher):
        """Python vs C++ bubble sort should be HIGH confidence."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', CPP_BUBBLE_SORT, 'cpp')
        
        assert result.plagiarism_detected is True
        assert result.confidence == 'high', \
            f"Expected 'high', got '{result.confidence}' (structural: {result.structural_similarity:.1%})"
        assert len(result.matching_patterns) > 0
    
    def test_java_vs_cpp_bubble_sort(self, hasher):
        """Java vs C++ bubble sort should be HIGH confidence."""
        result = hasher.compare(JAVA_BUBBLE_SORT, 'java', CPP_BUBBLE_SORT, 'cpp')
        
        assert result.plagiarism_detected is True
        assert result.confidence == 'high', \
            f"Expected 'high', got '{result.confidence}' (structural: {result.structural_similarity:.1%})"


# ============================================================================
# ComparisonResult Tests
# ============================================================================

class TestComparisonResult:
    """Test ComparisonResult dataclass structure."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_result_type(self, hasher):
        """Result should be ComparisonResult."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', JAVA_BUBBLE_SORT, 'java')
        assert isinstance(result, ComparisonResult)
    
    def test_result_has_all_fields(self, hasher):
        """Result should have all expected fields."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', JAVA_BUBBLE_SORT, 'java')
        
        assert hasattr(result, 'similarity')
        assert hasattr(result, 'plagiarism_detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'syntactic_similarity')
        assert hasattr(result, 'hamming_distance')
        assert hasattr(result, 'structural_similarity')
        assert hasattr(result, 'matching_patterns')
        assert hasattr(result, 'pattern_match_ratio')
        assert hasattr(result, 'lang1')
        assert hasattr(result, 'lang2')
        assert hasattr(result, 'method_used')
    
    def test_result_types_correct(self, hasher):
        """Result fields should have correct types."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', JAVA_BUBBLE_SORT, 'java')
        
        assert isinstance(result.similarity, float)
        assert isinstance(result.plagiarism_detected, bool)
        assert isinstance(result.confidence, str)
        assert isinstance(result.syntactic_similarity, float)
        assert isinstance(result.hamming_distance, int)
        assert isinstance(result.structural_similarity, float)
        assert isinstance(result.matching_patterns, list)
        assert isinstance(result.pattern_match_ratio, str)
        assert isinstance(result.lang1, str)
        assert isinstance(result.lang2, str)
        assert isinstance(result.method_used, str)
    
    def test_similarity_in_range(self, hasher):
        """Similarities should be between 0 and 1."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', PYTHON_QUICK_SORT, 'python')
        
        assert 0.0 <= result.similarity <= 1.0
        assert 0.0 <= result.syntactic_similarity <= 1.0
        assert 0.0 <= result.structural_similarity <= 1.0
    
    def test_hamming_distance_in_range(self, hasher):
        """Hamming distance should be between 0 and 256."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', PYTHON_QUICK_SORT, 'python')
        assert 0 <= result.hamming_distance <= 256
    
    def test_confidence_valid_value(self, hasher):
        """Confidence should be high, medium, or low."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', JAVA_BUBBLE_SORT, 'java')
        assert result.confidence in ['high', 'medium', 'low']
    
    def test_method_used_valid_value(self, hasher):
        """Method used should be syntactic or structural."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', JAVA_BUBBLE_SORT, 'java')
        assert result.method_used in ['syntactic', 'structural']


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge case handling."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_empty_code_both(self, hasher):
        """Both empty codes should not crash."""
        result = hasher.compare("", 'python', "", 'python')
        assert result.confidence == 'low' or result.similarity >= 0.0
    
    def test_empty_code_one_side(self, hasher):
        """One empty code should not crash."""
        result = hasher.compare(PYTHON_BUBBLE_SORT, 'python', "", 'python')
        assert 0.0 <= result.similarity <= 1.0
    
    def test_minimal_code(self, hasher):
        """Minimal single-line code works."""
        result = hasher.compare("x = 1", 'python', "y = 2", 'python')
        assert 0.0 <= result.similarity <= 1.0
    
    def test_whitespace_only(self, hasher):
        """Whitespace-only code doesn't crash."""
        result = hasher.compare("   \n\n  ", 'python', "  \t\n", 'python')
        assert 0.0 <= result.similarity <= 1.0
    
    def test_compare_hashes_different_length(self, hasher):
        """Comparing hashes of different lengths raises error."""
        hash1 = np.array([0, 1, 0, 1])
        hash2 = np.array([1, 0, 1])
        
        with pytest.raises(ValueError, match="Hashes must be the same length"):
            hasher.compare_hashes(hash1, hash2)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
