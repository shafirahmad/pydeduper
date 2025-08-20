"""Optimized database operations with advanced batching, indexing, and connection pooling."""

import sqlite3
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Iterator
from contextlib import contextmanager
from datetime import datetime
from queue import Queue, Empty
import json

from ..config import Config
from .models import FileInfo, FolderInfo, DuplicateGroup

logger = logging.getLogger(__name__)


class OptimizedConnectionPool:
    """High-performance connection pool with WAL mode and optimizations."""
    
    def __init__(self, db_path: Path, max_connections: int = 20):
        """Initialize optimized connection pool."""
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self._closed = False
        
        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize pool with optimized connections
        for _ in range(max_connections):
            conn = self._create_optimized_connection()
            if conn:
                self.connections.put(conn)
    
    def _create_optimized_connection(self) -> Optional[sqlite3.Connection]:
        """Create an optimized database connection."""
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode for better concurrency
                timeout=30.0
            )
            
            # Set optimal pragmas for performance
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
            conn.execute("PRAGMA page_size=4096")  # Optimal page size
            conn.execute("PRAGMA wal_autocheckpoint=1000")  # Auto-checkpoint every 1000 pages
            
            # Enable query optimizer
            conn.execute("PRAGMA optimize")
            
            conn.row_factory = sqlite3.Row
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create optimized connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        try:
            conn = self.connections.get(timeout=30)
            yield conn
        except Empty:
            # Pool exhausted, create temporary connection
            conn = self._create_optimized_connection()
            if not conn:
                raise RuntimeError("Failed to create database connection")
            yield conn
        finally:
            if conn and not self._closed:
                try:
                    self.connections.put_nowait(conn)
                except:
                    conn.close()
    
    def execute_batch(self, query: str, data: List[Tuple], batch_size: int = 10000):
        """Execute batch operations with optimal batching."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Use transaction for batch operations
            cursor.execute("BEGIN IMMEDIATE")
            
            try:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    cursor.executemany(query, batch)
                
                cursor.execute("COMMIT")
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
    
    def close_all(self):
        """Close all connections and checkpoint WAL."""
        self._closed = True
        
        # Checkpoint WAL before closing
        try:
            with self.connections.get(timeout=1) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except:
            pass
        
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()
            except:
                break


class OptimizedDatabase:
    """Database with advanced optimization techniques."""
    
    def __init__(self, config: Config = None):
        """Initialize optimized database."""
        self.config = config or Config()
        self.db_path = self.config.DEFAULT_DB_PATH
        self.pool = OptimizedConnectionPool(self.db_path)
        
        # Prepared statement cache
        self._prepared_statements = {}
        
        # Statistics
        self.stats = {
            'inserts': 0,
            'updates': 0,
            'queries': 0,
            'total_time': 0.0
        }
    
    def initialize_optimized(self):
        """Create optimized database schema with advanced indexes."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables with optimized schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    parent_path TEXT,
                    num_files INTEGER DEFAULT 0,
                    total_size INTEGER DEFAULT 0,
                    scan_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) WITHOUT ROWID
            ''')
            
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
                    is_duplicate INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_folder_id) REFERENCES folders(id)
                ) WITHOUT ROWID
            ''')
            
            # Create optimized compound indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_size_hash ON files(size, hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_quick_hash ON files(quick_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_parent_size ON files(parent_folder_id, size)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_duplicate ON files(is_duplicate) WHERE is_duplicate = 1')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_path)')
            
            # Create covering indexes for common queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_files_covering 
                ON files(size, hash, path, name)
            ''')
            
            # Create partial indexes for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_files_large 
                ON files(size, hash) 
                WHERE size > 1048576
            ''')
            
            # Create triggers for auto-updating timestamps
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_folders_timestamp 
                AFTER UPDATE ON folders
                BEGIN
                    UPDATE folders SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END
            ''')
            
            # Create helper table for duplicate tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS duplicate_groups (
                    group_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    file_count INTEGER NOT NULL,
                    total_size INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_duplicate_groups_hash ON duplicate_groups(hash)')
            
            # Analyze tables for query optimizer
            cursor.execute("ANALYZE")
            
            conn.commit()
            logger.info("Optimized database initialized successfully")
    
    def bulk_insert_files(self, files: List[FileInfo], use_transaction: bool = True) -> int:
        """Bulk insert files with optimal batching."""
        if not files:
            return 0
        
        start_time = time.time()
        inserted = 0
        
        # Prepare data for bulk insert
        data = [
            (
                str(f.path),
                f.name,
                f.size,
                f.hash,
                f.quick_hash,
                f.parent_folder_id,
                f.last_modified.isoformat() if f.last_modified else None
            )
            for f in files
        ]
        
        query = '''
            INSERT OR REPLACE INTO files 
            (path, name, size, hash, quick_hash, parent_folder_id, last_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        
        try:
            self.pool.execute_batch(query, data, batch_size=10000)
            inserted = len(data)
            
            self.stats['inserts'] += inserted
            self.stats['total_time'] += time.time() - start_time
            
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
        
        return inserted
    
    def bulk_update_hashes(self, hash_updates: List[Tuple[str, str]]) -> int:
        """Bulk update file hashes."""
        if not hash_updates:
            return 0
        
        start_time = time.time()
        updated = 0
        
        query = 'UPDATE files SET hash = ? WHERE path = ?'
        
        # Reorder tuples for the query
        data = [(hash_val, path) for path, hash_val in hash_updates]
        
        try:
            self.pool.execute_batch(query, data, batch_size=10000)
            updated = len(data)
            
            self.stats['updates'] += updated
            self.stats['total_time'] += time.time() - start_time
            
        except Exception as e:
            logger.error(f"Bulk update failed: {e}")
        
        return updated
    
    def find_duplicates_optimized(self, min_size: int = 0) -> List[DuplicateGroup]:
        """Find duplicates using optimized queries."""
        start_time = time.time()
        
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Use optimized query with covering index
            query = '''
                WITH duplicate_hashes AS (
                    SELECT hash, size, COUNT(*) as count
                    FROM files
                    WHERE hash IS NOT NULL 
                    AND size > ?
                    GROUP BY hash, size
                    HAVING COUNT(*) > 1
                )
                SELECT f.path, f.name, f.size, f.hash, f.last_modified
                FROM files f
                INNER JOIN duplicate_hashes d ON f.hash = d.hash AND f.size = d.size
                ORDER BY f.size DESC, f.hash, f.path
            '''
            
            cursor.execute(query, (min_size,))
            
            # Group results into DuplicateGroup objects
            groups = {}
            for row in cursor.fetchall():
                hash_key = row['hash']
                if hash_key not in groups:
                    groups[hash_key] = DuplicateGroup(
                        hash=hash_key,
                        size=row['size'],
                        files=[]
                    )
                
                groups[hash_key].files.append(FileInfo(
                    path=Path(row['path']),
                    name=row['name'],
                    size=row['size'],
                    hash=row['hash'],
                    last_modified=datetime.fromisoformat(row['last_modified']) if row['last_modified'] else None
                ))
            
            self.stats['queries'] += 1
            self.stats['total_time'] += time.time() - start_time
            
            return list(groups.values())
    
    def get_files_by_size_range(self, min_size: int, max_size: int) -> Iterator[FileInfo]:
        """Get files within a size range using streaming."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Use streaming for large result sets
            cursor.execute('''
                SELECT path, name, size, hash, quick_hash, last_modified
                FROM files
                WHERE size BETWEEN ? AND ?
                ORDER BY size
            ''', (min_size, max_size))
            
            while True:
                rows = cursor.fetchmany(1000)  # Fetch in chunks
                if not rows:
                    break
                
                for row in rows:
                    yield FileInfo(
                        path=Path(row['path']),
                        name=row['name'],
                        size=row['size'],
                        hash=row['hash'],
                        quick_hash=row['quick_hash'],
                        last_modified=datetime.fromisoformat(row['last_modified']) if row['last_modified'] else None
                    )
    
    def create_duplicate_report(self) -> Dict[str, Any]:
        """Generate comprehensive duplicate report with statistics."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get duplicate statistics
            cursor.execute('''
                WITH dup_stats AS (
                    SELECT 
                        hash,
                        COUNT(*) as file_count,
                        SUM(size) as total_size,
                        size as file_size
                    FROM files
                    WHERE hash IS NOT NULL
                    GROUP BY hash, size
                    HAVING COUNT(*) > 1
                )
                SELECT 
                    COUNT(DISTINCT hash) as unique_duplicates,
                    SUM(file_count) as total_duplicate_files,
                    SUM(total_size) as total_duplicate_size,
                    SUM((file_count - 1) * file_size) as wasted_space
                FROM dup_stats
            ''')
            
            stats = cursor.fetchone()
            
            # Get size distribution
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN size < 1024 THEN '< 1KB'
                        WHEN size < 1048576 THEN '1KB - 1MB'
                        WHEN size < 10485760 THEN '1MB - 10MB'
                        WHEN size < 104857600 THEN '10MB - 100MB'
                        WHEN size < 1073741824 THEN '100MB - 1GB'
                        ELSE '> 1GB'
                    END as size_range,
                    COUNT(*) as count,
                    SUM(size) as total_size
                FROM files
                GROUP BY size_range
                ORDER BY MIN(size)
            ''')
            
            size_distribution = cursor.fetchall()
            
            return {
                'duplicate_stats': dict(stats) if stats else {},
                'size_distribution': [dict(row) for row in size_distribution],
                'database_stats': self.stats
            }
    
    def vacuum_database(self):
        """Optimize database file size and performance."""
        with self.pool.get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.execute("PRAGMA optimize")
    
    def close(self):
        """Close database connections."""
        self.pool.close_all()


class MemoryIndex:
    """In-memory index for ultra-fast lookups during scanning."""
    
    def __init__(self):
        self.size_index = {}  # size -> list of file paths
        self.hash_index = {}  # hash -> list of file paths
        self.path_index = {}  # path -> FileInfo
        self.lock = threading.Lock()
    
    def add_file(self, file_info: FileInfo):
        """Add file to indexes."""
        with self.lock:
            # Size index
            if file_info.size not in self.size_index:
                self.size_index[file_info.size] = []
            self.size_index[file_info.size].append(str(file_info.path))
            
            # Hash index
            if file_info.hash:
                if file_info.hash not in self.hash_index:
                    self.hash_index[file_info.hash] = []
                self.hash_index[file_info.hash].append(str(file_info.path))
            
            # Path index
            self.path_index[str(file_info.path)] = file_info
    
    def get_files_by_size(self, size: int) -> List[str]:
        """Get all files with specific size."""
        return self.size_index.get(size, [])
    
    def get_files_by_hash(self, hash_value: str) -> List[str]:
        """Get all files with specific hash."""
        return self.hash_index.get(hash_value, [])
    
    def has_duplicate_size(self, size: int) -> bool:
        """Check if size has potential duplicates."""
        return len(self.size_index.get(size, [])) > 1
    
    def clear(self):
        """Clear all indexes."""
        with self.lock:
            self.size_index.clear()
            self.hash_index.clear()
            self.path_index.clear()