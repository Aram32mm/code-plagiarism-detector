"""Database module for storing and querying reference code hashes."""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import threading


class CodeHashDatabase:
    """Thread-safe SQLite database for storing and querying code hashes."""
    
    def __init__(self, db_path: str = "code_hashes.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self._local = threading.local()
        self._create_tables()
    
    @property
    def conn(self):
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.conn
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source VARCHAR(255) NOT NULL,
                file_path TEXT NOT NULL,
                language VARCHAR(50) NOT NULL,
                code TEXT NOT NULL,
                hash_value TEXT NOT NULL,
                patterns TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, file_path)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON code_hashes(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_language ON code_hashes(language)")
        
        self.conn.commit()
    
    def add_hash(self, source: str, file_path: str, language: str,
                 code: str, hash_value: np.ndarray, patterns: List[str] = None,
                 metadata: Optional[Dict] = None) -> int:
        """
        Add code with hash and patterns to the database.
        
        Args:
            source: Source identifier (e.g., 'leetcode', 'homework1')
            file_path: Original file path
            language: Programming language
            code: Original source code
            hash_value: 256-bit hash as numpy array
            patterns: Control flow patterns list
            metadata: Optional metadata dictionary
            
        Returns:
            ID of inserted record
        """
        cursor = self.conn.cursor()
        
        hash_str = ','.join(map(str, hash_value.astype(int).tolist()))
        patterns_str = '|'.join(patterns) if patterns else ''
        metadata_str = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO code_hashes 
            (source, file_path, language, code, hash_value, patterns, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source, file_path, language, code, hash_str, patterns_str, metadata_str))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def find_similar_hash(self, query_hash: np.ndarray, threshold: float = 0.5,
                          source_filter: Optional[str] = None,
                          language_filter: Optional[str] = None,
                          limit: int = 100) -> List[Dict]:
        """
        Find similar code using hash comparison (syntactic).
        
        Args:
            query_hash: Hash to search for
            threshold: Minimum similarity threshold (0.0-1.0)
            source_filter: Optional filter by source
            language_filter: Optional filter by language
            limit: Maximum number of results
            
        Returns:
            List of matching records with similarity scores
        """
        cursor = self.conn.cursor()
        
        query = "SELECT id, source, file_path, language, code, hash_value, patterns, metadata FROM code_hashes"
        params = []
        conditions = []
        
        if source_filter:
            conditions.append("source = ?")
            params.append(source_filter)
        if language_filter:
            conditions.append("language = ?")
            params.append(language_filter)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            record_id, source, file_path, language, code, hash_str, patterns_str, metadata_str = row
            
            stored_hash = np.array([int(x) for x in hash_str.split(',')], dtype=np.uint8)
            hamming_distance = np.sum(query_hash != stored_hash)
            similarity = 1.0 - (hamming_distance / len(query_hash))
            
            if similarity >= threshold:
                results.append({
                    'id': record_id,
                    'source': source,
                    'file_path': file_path,
                    'language': language,
                    'code': code,
                    'patterns': patterns_str.split('|') if patterns_str else [],
                    'similarity': float(similarity),
                    'hamming_distance': int(hamming_distance),
                    'match_type': 'syntactic',
                    'metadata': json.loads(metadata_str) if metadata_str else {}
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def find_similar_structural(self, query_patterns: List[str], threshold: float = 0.5,
                                source_filter: Optional[str] = None,
                                language_filter: Optional[str] = None,
                                limit: int = 100) -> List[Dict]:
        """
        Find similar code using structural pattern comparison.
        
        Args:
            query_patterns: Control flow patterns to search for
            threshold: Minimum similarity threshold (0.0-1.0)
            source_filter: Optional filter by source
            language_filter: Optional filter by language
            limit: Maximum number of results
            
        Returns:
            List of matching records with similarity scores
        """
        cursor = self.conn.cursor()
        
        query = "SELECT id, source, file_path, language, code, patterns, metadata FROM code_hashes"
        params = []
        conditions = []
        
        if source_filter:
            conditions.append("source = ?")
            params.append(source_filter)
        if language_filter:
            conditions.append("language = ?")
            params.append(language_filter)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        
        # Generate query shingles
        query_shingles = set(self._generate_shingles(query_patterns, k=2))
        
        results = []
        for row in cursor.fetchall():
            record_id, source, file_path, language, code, patterns_str, metadata_str = row
            
            if not patterns_str:
                continue
            
            stored_patterns = patterns_str.split('|')
            stored_shingles = set(self._generate_shingles(stored_patterns, k=2))
            
            # Jaccard similarity
            intersection = query_shingles & stored_shingles
            union = query_shingles | stored_shingles
            similarity = len(intersection) / len(union) if union else 0.0
            
            if similarity >= threshold:
                results.append({
                    'id': record_id,
                    'source': source,
                    'file_path': file_path,
                    'language': language,
                    'code': code,
                    'patterns': stored_patterns,
                    'similarity': float(similarity),
                    'matching_patterns': sorted(list(intersection)),
                    'match_type': 'structural',
                    'metadata': json.loads(metadata_str) if metadata_str else {}
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def find_similar(self, query_hash: np.ndarray, query_patterns: List[str],
                     threshold: float = 0.5,
                     source_filter: Optional[str] = None,
                     language_filter: Optional[str] = None,
                     limit: int = 100) -> List[Dict]:
        """
        Find similar code using BOTH hash and structural comparison.
        Returns the best match from either method.
        
        Args:
            query_hash: Hash to search for
            query_patterns: Control flow patterns to search for
            threshold: Minimum similarity threshold
            source_filter: Optional filter by source
            language_filter: Optional filter by language
            limit: Maximum number of results
            
        Returns:
            List of matching records with best similarity scores
        """
        # Get results from both methods
        hash_results = self.find_similar_hash(
            query_hash, threshold, source_filter, language_filter, limit * 2
        )
        structural_results = self.find_similar_structural(
            query_patterns, threshold, source_filter, language_filter, limit * 2
        )
        
        # Merge results by file_path, keeping best similarity
        merged = {}
        
        for r in hash_results:
            key = (r['source'], r['file_path'])
            merged[key] = {
                'id': r['id'],
                'source': r['source'],
                'file_path': r['file_path'],
                'language': r['language'],
                'code': r['code'],
                'patterns': r['patterns'],
                'syntactic_similarity': r['similarity'],
                'structural_similarity': 0.0,
                'similarity': r['similarity'],
                'match_type': 'syntactic',
                'metadata': r['metadata']
            }
        
        for r in structural_results:
            key = (r['source'], r['file_path'])
            if key in merged:
                merged[key]['structural_similarity'] = r['similarity']
                merged[key]['matching_patterns'] = r.get('matching_patterns', [])
                # Use best similarity
                if r['similarity'] > merged[key]['syntactic_similarity']:
                    merged[key]['similarity'] = r['similarity']
                    merged[key]['match_type'] = 'structural'
            else:
                merged[key] = {
                    'id': r['id'],
                    'source': r['source'],
                    'file_path': r['file_path'],
                    'language': r['language'],
                    'code': r['code'],
                    'patterns': r['patterns'],
                    'syntactic_similarity': 0.0,
                    'structural_similarity': r['similarity'],
                    'similarity': r['similarity'],
                    'match_type': 'structural',
                    'matching_patterns': r.get('matching_patterns', []),
                    'metadata': r['metadata']
                }
        
        results = list(merged.values())
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def get_code(self, record_id: int) -> Optional[Dict]:
        """Get full record including code by ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, source, file_path, language, code, hash_value, patterns, metadata
            FROM code_hashes WHERE id = ?
        """, (record_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        record_id, source, file_path, language, code, hash_str, patterns_str, metadata_str = row
        
        return {
            'id': record_id,
            'source': source,
            'file_path': file_path,
            'language': language,
            'code': code,
            'hash': np.array([int(x) for x in hash_str.split(',')], dtype=np.uint8),
            'patterns': patterns_str.split('|') if patterns_str else [],
            'metadata': json.loads(metadata_str) if metadata_str else {}
        }
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM code_hashes")
        total_hashes = cursor.fetchone()[0]
        
        cursor.execute("SELECT language, COUNT(*) FROM code_hashes GROUP BY language")
        by_language = dict(cursor.fetchall())
        
        cursor.execute("SELECT source, COUNT(*) FROM code_hashes GROUP BY source")
        by_source = dict(cursor.fetchall())
        
        return {
            'total_hashes': total_hashes,
            'by_language': by_language,
            'by_source': by_source
        }
    
    def clear(self):
        """Clear all hashes from database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM code_hashes")
        self.conn.commit()
    
    def _generate_shingles(self, features: List[str], k: int = 2) -> List[str]:
        """Generate k-shingles from features."""
        if not features:
            return []
        if len(features) < k:
            return [' '.join(features)] if features else []
        return [' '.join(features[i:i+k]) for i in range(len(features) - k + 1)]
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_code_bank(code_bank_path: str, hasher, db: CodeHashDatabase,
                   force_reload: bool = False) -> int:
    """
    Load code_bank folder into database on startup.
    
    Args:
        code_bank_path: Path to code_bank folder
        hasher: CodeHasher instance
        db: CodeHashDatabase instance
        force_reload: If True, clear DB and reload all
        
    Returns:
        Number of files loaded
    """
    path = Path(code_bank_path)
    
    if not path.exists():
        return 0
    
    if not force_reload:
        stats = db.get_stats()
        if stats['total_hashes'] > 0:
            return stats['total_hashes']
    
    if force_reload:
        db.clear()
    
    extensions = {
        '.py': 'python',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.c': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp'
    }
    
    loaded = 0
    
    for file in path.rglob('*'):
        if not file.is_file():
            continue
        
        ext = file.suffix.lower()
        if ext not in extensions:
            continue
        
        try:
            language = extensions[ext]
            
            # Read code
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Determine source from folder structure
            relative = file.relative_to(path)
            parts = relative.parts
            source = 'code_bank'
            
            # Generate hash and patterns
            hash_value = hasher.hash_code(code, language)
            patterns = hasher.debug_patterns(code, language)['control_flow']
            
            # Metadata
            metadata = {
                'filename': file.name,
                'folder': str(relative.parent),
                'category': '/'.join(parts[:-1]),
                'size': len(code),
                'loaded_at': datetime.now().isoformat()
            }
            
            db.add_hash(
                source=source,
                file_path=str(relative),
                language=language,
                code=code,
                hash_value=hash_value,
                patterns=patterns,
                metadata=metadata
            )
            
            loaded += 1
            
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    return loaded
