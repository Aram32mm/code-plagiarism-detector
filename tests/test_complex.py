"""Advanced tests for cross-language plagiarism detection."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from code_plagiarism_detector import CodeHasher, ComparisonResult


class TestEdgeCases:
    """Edge case handling."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_empty_code(self, hasher):
        """Empty code should return 100% similarity."""
        result = hasher.compare('', 'python', '', 'python')
        assert result.similarity == 1.0
        assert result.confidence == 'high'
    
    def test_empty_vs_non_empty(self, hasher):
        """Empty vs non-empty should have low similarity."""
        code = '''
class Solution:
    def test(self):
        for i in range(10):
            if i > 5:
                print(i)
'''
        result = hasher.compare('', 'python', code, 'python')
        assert result.structural_similarity == 0.0
    
    def test_minimal_code(self, hasher):
        """Minimal code with no control flow."""
        py = 'x = 1'
        java = 'int x = 1;'
        result = hasher.compare(py, 'python', java, 'java')
        assert result.similarity == 1.0  # Both have no control flow
    
    def test_whitespace_ignored(self, hasher):
        """Whitespace and comments should not affect similarity."""
        clean = '''
class Solution:
    def add(self, a, b):
        if a > 0:
            return a + b
        return b
'''
        messy = '''
class Solution:
    # This is a comment
    def add(self, a, b):
        # Another comment
        if a > 0:   # inline comment
            return a + b
        
        
        return b
'''
        result = hasher.compare(clean, 'python', messy, 'python')
        assert result.similarity == 1.0
        assert result.structural_similarity == 1.0


class TestLoopPatterns:
    """Test different loop patterns."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_single_loop_cross_language(self, hasher):
        """Single loop should match across languages."""
        py = '''
class Solution:
    def sum_array(self, arr):
        total = 0
        for i in range(len(arr)):
            total += arr[i]
        return total
'''
        java = '''
public class Solution {
    public int sumArray(int[] arr) {
        int total = 0;
        for (int i = 0; i < arr.length; i++) {
            total += arr[i];
        }
        return total;
    }
}
'''
        result = hasher.compare(py, 'python', java, 'java')
        assert result.structural_similarity == 1.0
        assert result.confidence == 'high'
    
    def test_while_vs_for_same_pattern(self, hasher):
        """While and for loops should both map to LOOP."""
        for_loop = '''
class Solution:
    def countdown(self, n):
        for i in range(n, 0, -1):
            print(i)
'''
        while_loop = '''
class Solution:
    def countdown(self, n):
        while n > 0:
            print(n)
            n -= 1
'''
        result = hasher.compare(for_loop, 'python', while_loop, 'python')
        assert result.structural_similarity == 1.0
    
    def test_nested_loops(self, hasher):
        """Double nested loops should match."""
        py = '''
class Solution:
    def matrix_sum(self, matrix):
        total = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                total += matrix[i][j]
        return total
'''
        cpp = '''
class Solution {
public:
    int matrixSum(vector<vector<int>>& matrix) {
        int total = 0;
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[0].size(); j++) {
                total += matrix[i][j];
            }
        }
        return total;
    }
};
'''
        result = hasher.compare(py, 'python', cpp, 'cpp')
        assert result.structural_similarity == 1.0
        assert result.confidence == 'high'
    
    def test_triple_nested_loops(self, hasher):
        """Triple nested loops with condition."""
        py = '''
class Solution:
    def matrix_multiply(self, a, b):
        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(b)):
                    if a[i][k] != 0:
                        c[i][j] += a[i][k] * b[k][j]
'''
        cpp = '''
class Solution {
public:
    void matrixMultiply(vector<vector<int>>& a, vector<vector<int>>& b) {
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b[0].size(); j++) {
                for (int k = 0; k < b.size(); k++) {
                    if (a[i][k] != 0) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    }
};
'''
        result = hasher.compare(py, 'python', cpp, 'cpp')
        assert result.structural_similarity == 1.0
        assert result.confidence == 'high'


class TestConditionPatterns:
    """Test different condition patterns."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_single_condition(self, hasher):
        """Single condition should match."""
        py = '''
class Solution:
    def absolute(self, x):
        if x < 0:
            return -x
        return x
'''
        java = '''
public class Solution {
    public int absolute(int x) {
        if (x < 0) {
            return -x;
        }
        return x;
    }
}
'''
        result = hasher.compare(py, 'python', java, 'java')
        assert result.structural_similarity == 1.0
    
    def test_multiple_conditions(self, hasher):
        """Multiple sequential conditions."""
        py = '''
class Solution:
    def grade(self, score):
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        return "F"
'''
        java = '''
public class Solution {
    public String grade(int score) {
        if (score >= 90) {
            return "A";
        }
        if (score >= 80) {
            return "B";
        }
        if (score >= 70) {
            return "C";
        }
        return "F";
    }
}
'''
        result = hasher.compare(py, 'python', java, 'java')
        assert result.structural_similarity == 1.0
        assert result.confidence == 'high'


class TestDifferentAlgorithms:
    """Different algorithms should NOT match."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_bubble_vs_binary_search(self, hasher):
        """Bubble sort vs binary search should not match."""
        bubble = '''
class Solution:
    def bubble_sort(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
'''
        binary = '''
class Solution:
    def binary_search(self, arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
'''
        result = hasher.compare(bubble, 'python', binary, 'python')
        assert result.structural_similarity < 0.5
        assert result.confidence == 'low'
        assert result.plagiarism_detected is False
    
    def test_iterative_vs_recursive(self, hasher):
        """Iterative vs recursive should differ."""
        iterative = '''
class Solution:
    def factorial(self, n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
'''
        recursive = '''
class Solution:
    def factorial(self, n):
        if n <= 1:
            return 1
        return n * self.factorial(n - 1)
'''
        result = hasher.compare(iterative, 'python', recursive, 'python')
        assert result.structural_similarity < 0.5


class TestComplexAlgorithms:
    """Complex real-world algorithms."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_binary_search_cross_language(self, hasher):
        """Binary search across all 3 languages."""
        py = '''
class Solution:
    def search(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
'''
        java = '''
public class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }
}
'''
        cpp = '''
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }
};
'''
        # Python vs Java
        r1 = hasher.compare(py, 'python', java, 'java')
        assert r1.structural_similarity == 1.0
        assert r1.confidence == 'high'
        
        # Python vs C++
        r2 = hasher.compare(py, 'python', cpp, 'cpp')
        assert r2.structural_similarity == 1.0
        assert r2.confidence == 'high'
        
        # Java vs C++
        r3 = hasher.compare(java, 'java', cpp, 'cpp')
        assert r3.structural_similarity == 1.0
        assert r3.confidence == 'high'
    
    def test_merge_sort_cross_language(self, hasher):
        """Merge sort with multiple methods."""
        py = '''
class Solution:
    def mergeSort(self, arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = self.mergeSort(arr[:mid])
        right = self.mergeSort(arr[mid:])
        return self.merge(left, right)

    def merge(self, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        while i < len(left):
            result.append(left[i])
            i += 1
        while j < len(right):
            result.append(right[j])
            j += 1
        return result
'''
        java = '''
public class Solution {
    public int[] mergeSort(int[] arr) {
        if (arr.length <= 1) {
            return arr;
        }
        int mid = arr.length / 2;
        int[] left = mergeSort(Arrays.copyOfRange(arr, 0, mid));
        int[] right = mergeSort(Arrays.copyOfRange(arr, mid, arr.length));
        return merge(left, right);
    }

    private int[] merge(int[] left, int[] right) {
        int[] result = new int[left.length + right.length];
        int i = 0, j = 0, k = 0;
        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                result[k++] = left[i++];
            } else {
                result[k++] = right[j++];
            }
        }
        while (i < left.length) {
            result[k++] = left[i++];
        }
        while (j < right.length) {
            result[k++] = right[j++];
        }
        return result;
    }
}
'''
        result = hasher.compare(py, 'python', java, 'java')
        assert result.structural_similarity >= 0.6
        assert result.confidence == 'high'
    
    def test_dijkstra_cross_language(self, hasher):
        """Dijkstra's algorithm - complex nested structure."""
        py = '''
class Solution:
    def dijkstra(self, graph, start):
        n = len(graph)
        dist = [float("inf")] * n
        dist[start] = 0
        visited = [False] * n
        
        for _ in range(n):
            u = -1
            for i in range(n):
                if not visited[i]:
                    if u == -1 or dist[i] < dist[u]:
                        u = i
            
            if dist[u] == float("inf"):
                break
            
            visited[u] = True
            
            for v in range(n):
                if graph[u][v] > 0 and not visited[v]:
                    if dist[u] + graph[u][v] < dist[v]:
                        dist[v] = dist[u] + graph[u][v]
        
        return dist
'''
        cpp = '''
class Solution {
public:
    vector<int> dijkstra(vector<vector<int>>& graph, int start) {
        int n = graph.size();
        vector<int> dist(n, INT_MAX);
        dist[start] = 0;
        vector<bool> visited(n, false);
        
        for (int count = 0; count < n; count++) {
            int u = -1;
            for (int i = 0; i < n; i++) {
                if (!visited[i]) {
                    if (u == -1 || dist[i] < dist[u]) {
                        u = i;
                    }
                }
            }
            
            if (dist[u] == INT_MAX) {
                break;
            }
            
            visited[u] = true;
            
            for (int v = 0; v < n; v++) {
                if (graph[u][v] > 0 && !visited[v]) {
                    if (dist[u] + graph[u][v] < dist[v]) {
                        dist[v] = dist[u] + graph[u][v];
                    }
                }
            }
        }
        
        return dist;
    }
};
'''
        result = hasher.compare(py, 'python', cpp, 'cpp')
        assert result.structural_similarity >= 0.6
        assert result.confidence == 'high'
        assert result.plagiarism_detected is True


class TestLRUCache:
    """LRU Cache - complex data structure."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_lru_cache_cross_language(self, hasher):
        """LRU Cache implementation."""
        py = '''
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.val
        return -1

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]
'''
        java = '''
public class LRUCache {
    private int capacity;
    private Map<Integer, Node> cache;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.cache = new HashMap<>();
    }

    public int get(int key) {
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            remove(node);
            add(node);
            return node.val;
        }
        return -1;
    }

    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            remove(cache.get(key));
        }
        Node node = new Node(key, value);
        add(node);
        cache.put(key, node);
        if (cache.size() > capacity) {
            Node lru = head.next;
            remove(lru);
            cache.remove(lru.key);
        }
    }
}
'''
        result = hasher.compare(py, 'python', java, 'java')
        assert result.structural_similarity >= 0.5
        assert result.plagiarism_detected is True


class TestDebugInfo:
    """Test debug info is populated correctly."""
    
    @pytest.fixture
    def hasher(self):
        return CodeHasher()
    
    def test_debug_patterns(self, hasher):
        """Debug patterns should contain control flow info."""
        code = '''
class Solution:
    def test(self):
        for i in range(10):
            if i > 5:
                print(i)
'''
        debug = hasher.debug_patterns(code, 'python')
        
        assert 'features' in debug
        assert 'normalized' in debug
        assert 'control_flow' in debug
        assert len(debug['control_flow']) > 0
    
    def test_comparison_debug_info(self, hasher):
        """Comparison result should have debug info."""
        py = '''
class Solution:
    def test(self):
        for i in range(10):
            print(i)
'''
        java = '''
public class Solution {
    public void test() {
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
        }
    }
}
'''
        result = hasher.compare(py, 'python', java, 'java')
        
        assert 'patterns1' in result.debug_info
        assert 'patterns2' in result.debug_info
        assert result.debug_info['patterns1'] == ['LOOP:d0']
        assert result.debug_info['patterns2'] == ['LOOP:d0']


class TestConfigurability:
    """Test detector configurability."""
    
    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        from code_plagiarism_detector import DetectorConfig
        
        strict_config = DetectorConfig(
            high_confidence_threshold=0.80,
            medium_confidence_threshold=0.60,
            plagiarism_threshold=0.70
        )
        
        hasher = CodeHasher(config=strict_config)
        
        py = '''
class Solution:
    def test(self):
        for i in range(10):
            if i > 5:
                print(i)
'''
        java = '''
public class Solution {
    public void test() {
        for (int i = 0; i < 10; i++) {
            if (i > 5) {
                System.out.println(i);
            }
        }
    }
}
'''
        result = hasher.compare(py, 'python', java, 'java')
        
        # With stricter thresholds, same code might get different confidence
        assert result.structural_similarity == 1.0
    
    def test_update_patterns(self):
        """Patterns can be updated at runtime."""
        hasher = CodeHasher()
        
        # Add a new pattern type
        hasher.update_patterns('python', {
            'TRY': ['try_statement'],
        })
        
        assert 'TRY' in hasher.languages['python'].pattern_mappings


if __name__ == '__main__':
    pytest.main([__file__, '-v'])