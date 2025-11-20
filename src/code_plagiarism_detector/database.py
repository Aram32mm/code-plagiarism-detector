"""Database module for storing and querying reference code hashes."""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime


class CodeHashDatabase:
    """SQLite database for storing and querying code hashes."""
    
    def __init__(self, db_path: str = "code_hashes.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main hash storage table
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
        
        # Index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source 
            ON code_hashes(source)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_language 
            ON code_hashes(language)
        """)
        
        # Collections table for organizing code sources
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Collection membership
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_members (
                collection_id INTEGER,
                hash_id INTEGER,
                FOREIGN KEY(collection_id) REFERENCES collections(id),
                FOREIGN KEY(hash_id) REFERENCES code_hashes(id),
                PRIMARY KEY(collection_id, hash_id)
            )
        """)
        
        self.conn.commit()
    
    def add_hash(self, source: str, file_path: str, language: str, 
                 hash_value: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """
        Add a code hash to the database.
        
        Args:
            source: Source of the code (e.g., 'leetcode', 'github', 'student_submission')
            file_path: Original file path or identifier
            language: Programming language
            hash_value: 256-bit hash as numpy array
            metadata: Optional metadata dictionary
            
        Returns:
            ID of inserted record
        """
        cursor = self.conn.cursor()
        
        # Convert hash to comma-separated string for storage
        hash_str = ','.join(map(str, hash_value.astype(int).tolist()))
        
        # Convert metadata to JSON
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
        """
        Find similar code in the database.
        
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
        
        # Build query with filters
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
            
            # Parse hash
            stored_hash = np.array([int(x) for x in hash_str.split(',')], dtype=np.uint8)
            
            # Calculate similarity
            hamming_distance = np.sum(query_hash != stored_hash)
            similarity = 1.0 - (hamming_distance / len(query_hash))
            
            # Only include if above threshold
            if similarity >= threshold:
                metadata = json.loads(metadata_str) if metadata_str else {}
                
                results.append({
                    'id': record_id,
                    'source': source,
                    'file_path': file_path,
                    'language': language,
                    'similarity': float(similarity),
                    'hamming_distance': int(hamming_distance),
                    'metadata': metadata
                })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:limit]
    
    def create_collection(self, name: str, description: str = "") -> int:
        """Create a new collection."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO collections (name, description)
            VALUES (?, ?)
        """, (name, description))
        self.conn.commit()
        return cursor.lastrowid
    
    def add_to_collection(self, collection_name: str, hash_id: int):
        """Add a hash to a collection."""
        cursor = self.conn.cursor()
        
        # Get collection ID
        cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        collection_id = result[0]
        
        cursor.execute("""
            INSERT OR IGNORE INTO collection_members (collection_id, hash_id)
            VALUES (?, ?)
        """, (collection_id, hash_id))
        
        self.conn.commit()
    
    def get_collection_hashes(self, collection_name: str) -> List[Dict]:
        """Get all hashes in a collection."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT ch.id, ch.source, ch.file_path, ch.language, ch.hash_value, ch.metadata
            FROM code_hashes ch
            JOIN collection_members cm ON ch.id = cm.hash_id
            JOIN collections c ON cm.collection_id = c.id
            WHERE c.name = ?
        """, (collection_name,))
        
        results = []
        for row in cursor.fetchall():
            record_id, source, file_path, language, hash_str, metadata_str = row
            
            results.append({
                'id': record_id,
                'source': source,
                'file_path': file_path,
                'language': language,
                'hash_value': np.array([int(x) for x in hash_str.split(',')], dtype=np.uint8),
                'metadata': json.loads(metadata_str) if metadata_str else {}
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM code_hashes")
        total_hashes = cursor.fetchone()[0]
        
        cursor.execute("SELECT language, COUNT(*) FROM code_hashes GROUP BY language")
        by_language = dict(cursor.fetchall())
        
        cursor.execute("SELECT source, COUNT(*) FROM code_hashes GROUP BY source")
        by_source = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM collections")
        total_collections = cursor.fetchone()[0]
        
        return {
            'total_hashes': total_hashes,
            'by_language': by_language,
            'by_source': by_source,
            'total_collections': total_collections
        }
    
    def bulk_import_directory(self, directory: str, source: str, 
                             hasher, recursive: bool = True) -> int:
        """
        Import all code files from a directory.
        
        Args:
            directory: Directory path
            source: Source identifier (e.g., 'leetcode', 'github:username/repo')
            hasher: CodeHasher instance
            recursive: Whether to search recursively
            
        Returns:
            Number of files imported
        """
        path = Path(directory)
        extensions = ['.py', '.java', '.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']
        
        if recursive:
            files = [f for f in path.rglob('*') if f.suffix in extensions and f.is_file()]
        else:
            files = [f for f in path.glob('*') if f.suffix in extensions and f.is_file()]
        
        imported = 0
        for file in files:
            try:
                hash_value = hasher.hash_file(str(file))
                
                metadata = {
                    'file_size': file.stat().st_size,
                    'imported_at': datetime.now().isoformat()
                }
                
                self.add_hash(
                    source=source,
                    file_path=str(file.relative_to(path)),
                    language=hasher._detect_language(str(file)),
                    hash_value=hash_value,
                    metadata=metadata
                )
                
                imported += 1
            except Exception as e:
                print(f"Warning: Could not process {file}: {e}")
        
        return imported
    
    def export_to_json(self, output_file: str):
        """Export database to JSON file."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM code_hashes")
        
        records = []
        for row in cursor.fetchall():
            record_id, source, file_path, language, hash_str, metadata_str, created_at = row
            
            records.append({
                'id': record_id,
                'source': source,
                'file_path': file_path,
                'language': language,
                'hash': hash_str,
                'metadata': json.loads(metadata_str) if metadata_str else {},
                'created_at': created_at
            })
        
        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2)
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
