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
                hash_value TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, file_path)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON code_hashes(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_language ON code_hashes(language)")
        
        self.conn.commit()
    
    def add_hash(self, source: str, file_path: str, language: str, 
                 hash_value: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """Add a code hash to the database."""
        cursor = self.conn.cursor()
        
        hash_str = ','.join(map(str, hash_value.astype(int).tolist()))
        metadata_str = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO code_hashes 
            (source, file_path, language, hash_value, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (source, file_path, language, hash_str, metadata_str))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def find_similar(self, query_hash: np.ndarray, threshold: float = 0.85,
                    source_filter: Optional[str] = None,
                    language_filter: Optional[str] = None,
                    limit: int = 100) -> List[Dict]:
        """Find similar code in the database."""
        cursor = self.conn.cursor()
        
        query = "SELECT id, source, file_path, language, hash_value, metadata FROM code_hashes"
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
            record_id, source, file_path, language, hash_str, metadata_str = row
            
            stored_hash = np.array([int(x) for x in hash_str.split(',')], dtype=np.uint8)
            hamming_distance = np.sum(query_hash != stored_hash)
            similarity = 1.0 - (hamming_distance / len(query_hash))
            
            if similarity >= threshold:
                results.append({
                    'id': record_id,
                    'source': source,
                    'file_path': file_path,
                    'language': language,
                    'similarity': float(similarity),
                    'hamming_distance': int(hamming_distance),
                    'metadata': json.loads(metadata_str) if metadata_str else {}
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
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
            
            relative = file.relative_to(path)
            parts = relative.parts
            
            if len(parts) >= 2:
                source = parts[1] if len(parts) > 2 else parts[0]
            else:
                source = 'code_bank'
            
            hash_value = hasher.hash_file(str(file))
            
            metadata = {
                'filename': file.name,
                'category': '/'.join(parts[:-1]),
                'loaded_at': datetime.now().isoformat()
            }
            
            db.add_hash(
                source=source,
                file_path=str(relative),
                language=language,
                hash_value=hash_value,
                metadata=metadata
            )
            
            loaded += 1
            
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    return loaded
