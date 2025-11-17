"""Example usage of code plagiarism detector."""

from code_plagiarism_detector import CodeHasher


def main():
    """Demonstrate basic usage of CodeHasher."""
    # Initialize hasher
    hasher = CodeHasher()
    
    # Example 1: Hash code strings
    print("Example 1: Hashing code strings")
    print("-" * 50)
    
    code1 = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
    
    code2 = """
def sum_list(data):
    result = 0
    for value in data:
        result += value
    return result
"""
    
    hash1 = hasher.hash_code(code1, 'python')
    hash2 = hasher.hash_code(code2, 'python')
    
    print(f"Hash 1 (first 32 bits): {hash1[:32]}")
    print(f"Hash 2 (first 32 bits): {hash2[:32]}")
    
    similarity, hamming_dist = hasher.compare(hash1, hash2)
    print(f"\nSimilarity: {similarity:.2%}")
    print(f"Hamming Distance: {hamming_dist}/256")
    
    # Example 2: Hash files
    print("\n" + "=" * 50)
    print("Example 2: Detecting plagiarism")
    print("-" * 50)
    
    # Create sample files for demonstration
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Original file
        original_file = os.path.join(tmpdir, "original.py")
        with open(original_file, 'w') as f:
            f.write("""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""")
        
        # Plagiarized file (same structure, renamed variables)
        plagiarized_file = os.path.join(tmpdir, "plagiarized.py")
        with open(plagiarized_file, 'w') as f:
            f.write("""
def sort_items(items):
    length = len(items)
    for x in range(length):
        for y in range(0, length - x - 1):
            if items[y] > items[y + 1]:
                items[y], items[y + 1] = items[y + 1], items[y]
    return items
""")
        
        # Different implementation
        different_file = os.path.join(tmpdir, "different.py")
        with open(different_file, 'w') as f:
            f.write("""
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""")
        
        # Hash all files
        hash_original = hasher.hash_file(original_file)
        hash_plagiarized = hasher.hash_file(plagiarized_file)
        hash_different = hasher.hash_file(different_file)
        
        # Compare original with plagiarized
        sim1, dist1 = hasher.compare(hash_original, hash_plagiarized)
        print(f"Original vs Plagiarized:")
        print(f"  Similarity: {sim1:.2%}")
        print(f"  Hamming Distance: {dist1}/256")
        
        # Compare original with different
        sim2, dist2 = hasher.compare(hash_original, hash_different)
        print(f"\nOriginal vs Different Algorithm:")
        print(f"  Similarity: {sim2:.2%}")
        print(f"  Hamming Distance: {dist2}/256")
        
        # Interpretation
        print("\n" + "=" * 50)
        print("Interpretation:")
        print("-" * 50)
        if sim1 > 0.7:
            print("⚠️  HIGH similarity detected between original and 'plagiarized' code!")
            print("   This suggests potential plagiarism.")
        if sim2 < 0.5:
            print("✓  LOW similarity between original and different implementation.")
            print("   These are clearly different algorithms.")
    
    # Example 3: Cross-language comparison
    print("\n" + "=" * 50)
    print("Example 3: Comparing different languages")
    print("-" * 50)
    
    python_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    
    java_code = """
public class Math {
    public int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }
}
"""
    
    cpp_code = """
int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
"""
    
    hash_python = hasher.hash_code(python_code, 'python')
    hash_java = hasher.hash_code(java_code, 'java')
    hash_cpp = hasher.hash_code(cpp_code, 'cpp')
    
    sim_py_java, _ = hasher.compare(hash_python, hash_java)
    sim_py_cpp, _ = hasher.compare(hash_python, hash_cpp)
    sim_java_cpp, _ = hasher.compare(hash_java, hash_cpp)
    
    print(f"Python vs Java:  {sim_py_java:.2%}")
    print(f"Python vs C++:   {sim_py_cpp:.2%}")
    print(f"Java vs C++:     {sim_java_cpp:.2%}")
    print("\nNote: Same algorithm in different languages shows structural similarity!")


if __name__ == "__main__":
    main()
