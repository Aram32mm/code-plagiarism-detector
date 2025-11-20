"""Basic usage examples for code plagiarism detector."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from code_plagiarism_detector import CodeHasher


def main():
    """Demonstrate basic usage of CodeHasher."""
    
    # Initialize the hasher
    hasher = CodeHasher()
    print("Code Plagiarism Detector - Basic Usage\n" + "="*50 + "\n")
    
    # Example 1: Hash a code string
    print("Example 1: Hashing Python code")
    print("-" * 50)
    
    code1 = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
    
    hash1 = hasher.hash_code(code1, 'python')
    print(f"Generated hash: {hash1[:16]}... (showing first 16 bits)")
    print(f"Hash length: {len(hash1)} bits\n")
    
    # Example 2: Compare similar code (variable names changed)
    print("Example 2: Comparing similar code with renamed variables")
    print("-" * 50)
    
    code2 = """
def bubble_sort(items):
    length = len(items)
    for x in range(length):
        for y in range(0, length - x - 1):
            if items[y] > items[y + 1]:
                items[y], items[y + 1] = items[y + 1], items[y]
    return items
"""
    
    hash2 = hasher.hash_code(code2, 'python')
    similarity, hamming_dist = hasher.compare(hash1, hash2)
    
    print(f"Similarity: {similarity:.2%}")
    print(f"Hamming Distance: {hamming_dist}/256")
    print(f"Assessment: {'PLAGIARISM DETECTED' if similarity > 0.85 else 'Similar structure detected' if similarity > 0.7 else 'Different code'}\n")
    
    # Example 3: Compare different algorithms
    print("Example 3: Comparing different algorithms")
    print("-" * 50)
    
    code3 = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""
    
    hash3 = hasher.hash_code(code3, 'python')
    similarity, hamming_dist = hasher.compare(hash1, hash3)
    
    print(f"Similarity: {similarity:.2%}")
    print(f"Hamming Distance: {hamming_dist}/256")
    print(f"Assessment: {'PLAGIARISM DETECTED' if similarity > 0.85 else 'Similar structure detected' if similarity > 0.7 else 'Different code'}\n")
    
    # Example 4: Java code
    print("Example 4: Hashing Java code")
    print("-" * 50)
    
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
    
    hash_java = hasher.hash_code(java_code, 'java')
    print(f"Java code hash generated: {hash_java[:16]}... (showing first 16 bits)")
    
    # Compare Python and Java implementations of same algorithm
    similarity, hamming_dist = hasher.compare(hash1, hash_java)
    print(f"\nStandard hash comparison (Python vs Java): {similarity:.2%}")
    print(f"Hamming Distance: {hamming_dist}/256")
    
    # NEW: Compare using cross-language method
    cross_sim, details = hasher.compare_cross_language(code1, 'python', java_code, 'java')
    print(f"\nCross-language comparison (Python vs Java): {cross_sim:.2%}")
    print(f"Matching patterns: {details['matching_shingles']}/{details['total_shingles']}")
    print(f"Assessment: Cross-language plagiarism {'DETECTED!' if cross_sim > 0.55 else 'not detected'}\n")
    
    # Example 5: C++ code
    print("Example 5: Hashing C++ code")
    print("-" * 50)
    
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
    
    hash_cpp = hasher.hash_code(cpp_code, 'cpp')
    similarity, hamming_dist = hasher.compare(hash1, hash_cpp)
    print(f"Standard hash comparison (Python vs C++): {similarity:.2%}")
    print(f"Hamming Distance: {hamming_dist}/256")
    
    # Cross-language comparison
    cross_sim, details = hasher.compare_cross_language(code1, 'python', cpp_code, 'cpp')
    print(f"\nCross-language comparison (Python vs C++): {cross_sim:.2%}")
    print(f"Matching patterns: {details['matching_shingles']}/{details['total_shingles']}\n")
    
    print("="*50)
    print("Examples completed successfully!")


if __name__ == '__main__':
    main()