"""Tests for CodeHasher class."""

import numpy as np
import pytest
from pathlib import Path

from code_plagiarism_detector import CodeHasher


class TestCodeHasher:
    """Test cases for CodeHasher class."""

    @pytest.fixture
    def hasher(self):
        """Create CodeHasher instance."""
        return CodeHasher()

    def test_init(self, hasher):
        """Test CodeHasher initialization."""
        assert hasher is not None
        assert 'python' in hasher.parsers
        assert 'java' in hasher.parsers
        assert 'cpp' in hasher.parsers

    def test_hash_code_python(self, hasher):
        """Test hashing Python code."""
        code = """
def hello():
    print("Hello, world!")
    
if __name__ == "__main__":
    hello()
"""
        hash_result = hasher.hash_code(code, 'python')
        assert hash_result is not None
        assert len(hash_result) == 256
        assert hash_result.dtype == np.uint8
        assert np.all((hash_result == 0) | (hash_result == 1))

    def test_hash_code_java(self, hasher):
        """Test hashing Java code."""
        code = """
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }
}
"""
        hash_result = hasher.hash_code(code, 'java')
        assert hash_result is not None
        assert len(hash_result) == 256
        assert hash_result.dtype == np.uint8

    def test_hash_code_cpp(self, hasher):
        """Test hashing C++ code."""
        code = """
#include <iostream>

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
"""
        hash_result = hasher.hash_code(code, 'cpp')
        assert hash_result is not None
        assert len(hash_result) == 256
        assert hash_result.dtype == np.uint8

    def test_hash_code_unsupported_language(self, hasher):
        """Test error handling for unsupported language."""
        code = "some code"
        with pytest.raises(ValueError, match="Unsupported language"):
            hasher.hash_code(code, 'rust')

    def test_hash_file_python(self, hasher, tmp_path):
        """Test hashing Python file."""
        # Create temporary Python file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
        
        hash_result = hasher.hash_file(str(test_file))
        assert hash_result is not None
        assert len(hash_result) == 256

    def test_hash_file_java(self, hasher, tmp_path):
        """Test hashing Java file."""
        test_file = tmp_path / "Test.java"
        test_file.write_text("""
public class Test {
    public int add(int a, int b) {
        return a + b;
    }
}
""")
        
        hash_result = hasher.hash_file(str(test_file))
        assert hash_result is not None
        assert len(hash_result) == 256

    def test_hash_file_cpp(self, hasher, tmp_path):
        """Test hashing C++ file."""
        test_file = tmp_path / "test.cpp"
        test_file.write_text("""
int add(int a, int b) {
    return a + b;
}
""")
        
        hash_result = hasher.hash_file(str(test_file))
        assert hash_result is not None
        assert len(hash_result) == 256

    def test_hash_file_not_found(self, hasher):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            hasher.hash_file("/nonexistent/file.py")

    def test_hash_file_unknown_extension(self, hasher, tmp_path):
        """Test error handling for unknown file extension."""
        test_file = tmp_path / "test.unknown"
        test_file.write_text("some code")
        
        with pytest.raises(ValueError, match="Could not detect language"):
            hasher.hash_file(str(test_file))

    def test_compare_identical_hashes(self, hasher):
        """Test comparing identical hashes."""
        code = "def foo(): pass"
        hash1 = hasher.hash_code(code, 'python')
        hash2 = hasher.hash_code(code, 'python')
        
        similarity, hamming_dist = hasher.compare(hash1, hash2)
        
        assert similarity == 1.0
        assert hamming_dist == 0

    def test_compare_similar_code(self, hasher):
        """Test comparing similar code snippets."""
        code1 = """
def calculate(x, y):
    result = x + y
    return result
"""
        code2 = """
def calculate(a, b):
    total = a + b
    return total
"""
        
        hash1 = hasher.hash_code(code1, 'python')
        hash2 = hasher.hash_code(code2, 'python')
        
        similarity, hamming_dist = hasher.compare(hash1, hash2)
        
        # Should be similar (high similarity score)
        assert 0.0 <= similarity <= 1.0
        assert 0 <= hamming_dist <= 256
        # Similar structure should have high similarity
        assert similarity > 0.7

    def test_compare_different_code(self, hasher):
        """Test comparing different code structures."""
        code1 = """
def simple_function():
    return 42
"""
        code2 = """
class ComplexClass:
    def __init__(self):
        self.data = []
    
    def process(self, items):
        for item in items:
            if item > 0:
                self.data.append(item)
        return self.data
"""
        
        hash1 = hasher.hash_code(code1, 'python')
        hash2 = hasher.hash_code(code2, 'python')
        
        similarity, hamming_dist = hasher.compare(hash1, hash2)
        
        # Should be different (lower similarity)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 0.8  # Different structures should have lower similarity

    def test_compare_different_lengths(self, hasher):
        """Test error handling for comparing hashes of different lengths."""
        hash1 = np.array([0, 1, 0, 1], dtype=np.uint8)
        hash2 = np.array([0, 1], dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Hash lengths must match"):
            hasher.compare(hash1, hash2)

    def test_feature_extraction_functions(self, hasher):
        """Test that function definitions are extracted."""
        code = """
def func1():
    pass

def func2(x):
    return x * 2

class MyClass:
    def method(self):
        pass
"""
        hash_result = hasher.hash_code(code, 'python')
        # Hash should be generated successfully
        assert len(hash_result) == 256

    def test_feature_extraction_control_flow(self, hasher):
        """Test that control flow structures are extracted."""
        code = """
def process(data):
    for item in data:
        if item > 0:
            while item > 10:
                item = item / 2
            try:
                result = item / 2
            except:
                pass
    return data
"""
        hash_result = hasher.hash_code(code, 'python')
        assert len(hash_result) == 256

    def test_feature_extraction_imports(self, hasher):
        """Test that imports are extracted."""
        code = """
import os
import sys
from pathlib import Path
from typing import List, Dict

def main():
    pass
"""
        hash_result = hasher.hash_code(code, 'python')
        assert len(hash_result) == 256

    def test_consistency(self, hasher):
        """Test that same code produces same hash."""
        code = "def test(): return 42"
        
        hash1 = hasher.hash_code(code, 'python')
        hash2 = hasher.hash_code(code, 'python')
        hash3 = hasher.hash_code(code, 'python')
        
        assert np.array_equal(hash1, hash2)
        assert np.array_equal(hash2, hash3)

    def test_language_detection(self, hasher):
        """Test language detection from file extensions."""
        assert hasher._detect_language('test.py') == 'python'
        assert hasher._detect_language('Test.java') == 'java'
        assert hasher._detect_language('test.cpp') == 'cpp'
        assert hasher._detect_language('test.cc') == 'cpp'
        assert hasher._detect_language('test.cxx') == 'cpp'
        assert hasher._detect_language('test.c') == 'cpp'
        assert hasher._detect_language('test.h') == 'cpp'
        assert hasher._detect_language('test.hpp') == 'cpp'
        assert hasher._detect_language('test.unknown') is None

    def test_plagiarism_detection_scenario(self, hasher, tmp_path):
        """Test a realistic plagiarism detection scenario."""
        # Original code
        original = tmp_path / "original.py"
        original.write_text("""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""")
        
        # Plagiarized version (renamed variables, same structure)
        plagiarized = tmp_path / "plagiarized.py"
        plagiarized.write_text("""
def sort_numbers(data):
    length = len(data)
    for x in range(length):
        for y in range(0, length - x - 1):
            if data[y] > data[y + 1]:
                data[y], data[y + 1] = data[y + 1], data[y]
    return data
""")
        
        # Different algorithm
        different = tmp_path / "different.py"
        different.write_text("""
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""")
        
        hash_original = hasher.hash_file(str(original))
        hash_plagiarized = hasher.hash_file(str(plagiarized))
        hash_different = hasher.hash_file(str(different))
        
        # Compare original with plagiarized (should be similar)
        sim_plagiarized, _ = hasher.compare(hash_original, hash_plagiarized)
        
        # Compare original with different (should be less similar)
        sim_different, _ = hasher.compare(hash_original, hash_different)
        
        # Plagiarized should be more similar than different algorithm
        assert sim_plagiarized > sim_different
        assert sim_plagiarized > 0.6  # Should detect similarity
