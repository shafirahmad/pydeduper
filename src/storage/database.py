"""Database operations for PyDeduper."""

import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
from datetime import datetime

from ..config import Config
from .models import FileInfo, FolderInfo, DuplicateGroup

logger = logging.getLogger(__name__)


class Database:
    """Handles all database operations for PyDeduper."""
    
    def __init__(self, config: Config = None):
        """Initialize database connection."""
        self.config = config or Config()
        self.db_path = self.config.DEFAULT_DB_PATH
        self.connection = None
        self._ensure_db_directory()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if self.connection is None:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
        try:
            yield self.connection
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        else:
            self.connection.commit()
    
    def initialize(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create folders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    parent_path TEXT,
                    num_files INTEGER DEFAULT 0,
                    total_size INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    hash TEXT,
                    quick_hash TEXT,
                    parent_folder_id INTEGER,
                    last_modified TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_folder_id) REFERENCES folders(id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_size ON files(size)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_quick_hash ON files(quick_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_path)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def clear_all(self):
        """Clear all data from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM files')
            cursor.execute('DELETE FROM folders')
            conn.commit()
            logger.info("Database cleared")
    
    def insert_folder(self, folder: FolderInfo) -> int:
        """
        Insert a folder into the database.
        
        Args:
            folder: FolderInfo object to insert
            
        Returns:
            ID of the inserted folder
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO folders (path, name, parent_path, num_files, total_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(folder.path),
                folder.name,
                str(folder.parent_path) if folder.parent_path else None,
                folder.num_files,
                folder.total_size
            ))
            return cursor.lastrowid
    
    def insert_file(self, file: FileInfo, folder_id: int = None) -> int:
        """
        Insert a file into the database.
        
        Args:
            file: FileInfo object to insert
            folder_id: ID of the parent folder
            
        Returns:
            ID of the inserted file
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO files 
                (path, name, size, hash, quick_hash, parent_folder_id, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(file.path),
                file.name,
                file.size,
                file.hash,
                file.quick_hash,
                folder_id or file.parent_folder_id,
                file.last_modified.isoformat() if file.last_modified else None
            ))
            return cursor.lastrowid
    
    def update_file_hash(self, file_id: int, hash_value: str):
        """Update the hash value for a file."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE files SET hash = ? WHERE id = ?',
                (hash_value, file_id)
            )
    
    def update_file_quick_hash(self, file_id: int, quick_hash: str):
        """Update the quick hash value for a file."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE files SET quick_hash = ? WHERE id = ?',
                (quick_hash, file_id)
            )
    
    def get_folder_by_path(self, path: Path) -> Optional[Dict[str, Any]]:
        """Get folder information by path."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM folders WHERE path = ?',
                (str(path),)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_files_by_folder(self, folder_id: int) -> List[Dict[str, Any]]:
        """Get all files in a folder."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM files WHERE parent_folder_id = ? ORDER BY name',
                (folder_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_files_by_size(self, size: int) -> List[Dict[str, Any]]:
        """Get all files with a specific size."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM files WHERE size = ? ORDER BY path',
                (size,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_files_by_hash(self, hash_value: str) -> List[Dict[str, Any]]:
        """Get all files with a specific hash."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM files WHERE hash = ? ORDER BY path',
                (hash_value,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_duplicate_sizes(self) -> List[Tuple[int, int]]:
        """
        Get file sizes that have duplicates.
        
        Returns:
            List of (size, count) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT size, COUNT(*) as count
                FROM files
                GROUP BY size
                HAVING count > 1
                ORDER BY size
            ''')
            return [(row['size'], row['count']) for row in cursor.fetchall()]
    
    def get_duplicate_hashes(self) -> List[Tuple[str, int, int]]:
        """
        Get hashes that have duplicates.
        
        Returns:
            List of (hash, size, count) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT hash, size, COUNT(*) as count
                FROM files
                WHERE hash IS NOT NULL AND hash != ''
                GROUP BY hash
                HAVING count > 1
                ORDER BY size DESC, hash
            ''')
            return [(row['hash'], row['size'], row['count']) for row in cursor.fetchall()]
    
    def get_folder_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about folders."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total counts
            cursor.execute('SELECT COUNT(*) as count FROM folders')
            total_folders = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM files')
            total_files = cursor.fetchone()['count']
            
            cursor.execute('SELECT SUM(size) as total FROM files')
            total_size = cursor.fetchone()['total'] or 0
            
            # Duplicate statistics
            cursor.execute('''
                SELECT COUNT(DISTINCT hash) as unique_hashes
                FROM files
                WHERE hash IS NOT NULL AND hash != ''
            ''')
            unique_hashes = cursor.fetchone()['unique_hashes']
            
            return {
                'total_folders': total_folders,
                'total_files': total_files,
                'total_size': total_size,
                'unique_hashes': unique_hashes
            }
    
    def calculate_folder_duplication(self, folder_id: int) -> float:
        """
        Calculate the duplication percentage for a folder.
        
        Args:
            folder_id: ID of the folder
            
        Returns:
            Percentage of files that are duplicated elsewhere
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total files in folder
            cursor.execute(
                'SELECT COUNT(*) as count FROM files WHERE parent_folder_id = ?',
                (folder_id,)
            )
            total_files = cursor.fetchone()['count']
            
            if total_files == 0:
                return 0.0
            
            # Count files with duplicates
            cursor.execute('''
                SELECT COUNT(DISTINCT f1.id) as duplicated_count
                FROM files f1
                INNER JOIN files f2 ON f1.hash = f2.hash
                WHERE f1.parent_folder_id = ?
                  AND f2.id != f1.id
                  AND f1.hash IS NOT NULL
                  AND f1.hash != ''
            ''', (folder_id,))
            
            duplicated_count = cursor.fetchone()['duplicated_count']
            
            return (duplicated_count / total_files) * 100
    
    def batch_insert_files(self, files: List[FileInfo], folder_id: int = None):
        """Batch insert multiple files for better performance."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = [
                (
                    str(f.path),
                    f.name,
                    f.size,
                    f.hash,
                    f.quick_hash,
                    folder_id or f.parent_folder_id,
                    f.last_modified.isoformat() if f.last_modified else None
                )
                for f in files
            ]
            
            cursor.executemany('''
                INSERT OR REPLACE INTO files 
                (path, name, size, hash, quick_hash, parent_folder_id, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data)
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")