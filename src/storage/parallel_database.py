"""Parallel database operations with batch processing and connection pooling."""

import sqlite3
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable
from contextlib import contextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import threading

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.current = 0
        
        def update(self, n=1):
            self.current += n
            print(f"\r{self.desc}: {self.current}/{self.total}", end="", flush=True)
        
        def set_description(self, desc):
            self.desc = desc
        
        def close(self):
            print()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

from ..config import Config
from .models import FileInfo, FolderInfo, DuplicateGroup
from .database import Database

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: Path, max_connections: int = 10):
        """
        Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database
            max_connections: Maximum number of concurrent connections
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self._closed = False
        
        # Pre-populate pool with connections
        for _ in range(max_connections):
            conn = self._create_connection()
            if conn:
                self.connections.put(conn)
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection."""
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            return conn
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        try:
            # Try to get connection from pool
            conn = self.connections.get(timeout=30)
            yield conn
        except Empty:
            # Pool exhausted, create temporary connection
            conn = self._create_connection()
            if not conn:
                raise RuntimeError("Failed to create database connection")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        else:
            if conn:
                conn.commit()
        finally:
            if conn and not self._closed:
                try:
                    # Return connection to pool if pool isn't full
                    self.connections.put_nowait(conn)
                except:
                    # Pool is full or closed, close this connection
                    conn.close()
    
    def close_all(self):
        """Close all connections in the pool."""
        self._closed = True
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()
            except:
                break


class BatchProcessor:
    """Handles batch processing of database operations."""
    
    def __init__(self, db_pool: ConnectionPool, batch_size: int = 1000):
        """
        Initialize batch processor.
        
        Args:
            db_pool: Connection pool
            batch_size: Number of items to process in each batch
        """
        self.db_pool = db_pool
        self.batch_size = batch_size
    
    def batch_insert_files(self, files: List[FileInfo], progress_callback: Optional[Callable] = None):
        """
        Insert files in batches with progress reporting.
        
        Args:
            files: List of FileInfo objects to insert
            progress_callback: Optional progress callback
        """
        if not files:
            return
        
        total_batches = (len(files) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(files), desc="Inserting files", unit='files', 
                 disable=progress_callback is not None) as pbar:
            
            for i in range(0, len(files), self.batch_size):
                batch = files[i:i + self.batch_size]
                
                with self.db_pool.get_connection() as conn:
                    cursor = conn.cursor()
                    
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
                        for f in batch
                    ]
                    
                    cursor.executemany('''
                        INSERT OR REPLACE INTO files 
                        (path, name, size, hash, quick_hash, parent_folder_id, last_modified)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', data)
                
                if progress_callback:
                    progress_callback(min(i + self.batch_size, len(files)), len(files))
                else:
                    pbar.update(len(batch))
    
    def batch_update_hashes(self, hash_updates: List[Tuple[int, str]], 
                           progress_callback: Optional[Callable] = None):
        """
        Update file hashes in batches.
        
        Args:
            hash_updates: List of (file_id, hash_value) tuples
            progress_callback: Optional progress callback
        """
        if not hash_updates:
            return
        
        with tqdm(total=len(hash_updates), desc="Updating hashes", unit='files',
                 disable=progress_callback is not None) as pbar:
            
            for i in range(0, len(hash_updates), self.batch_size):
                batch = hash_updates[i:i + self.batch_size]
                
                with self.db_pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.executemany(
                        'UPDATE files SET hash = ? WHERE id = ?',
                        [(hash_val, file_id) for file_id, hash_val in batch]
                    )
                
                if progress_callback:
                    progress_callback(min(i + self.batch_size, len(hash_updates)), len(hash_updates))
                else:
                    pbar.update(len(batch))
    
    def parallel_query_processing(self, queries: List[str], 
                                 max_workers: int = 4) -> List[List[Dict[str, Any]]]:
        """
        Execute multiple queries in parallel.
        
        Args:
            queries: List of SQL queries to execute
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of query results
        """
        results = [None] * len(queries)
        
        def execute_query(query_index: int, query: str):
            try:
                with self.db_pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    return query_index, [dict(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Error executing query {query_index}: {e}")
                return query_index, []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(execute_query, i, query): i 
                for i, query in enumerate(queries)
            }
            
            for future in as_completed(future_to_index):
                query_index, result = future.result()
                results[query_index] = result
        
        return results


class ParallelDatabase(Database):
    """Enhanced database with parallel operations and connection pooling."""
    
    def __init__(self, config: Config = None, max_connections: int = 10):
        """
        Initialize parallel database.
        
        Args:
            config: Configuration object
            max_connections: Maximum number of database connections
        """
        super().__init__(config)
        self.max_connections = max_connections
        self.connection_pool = None
        self.batch_processor = None
        self._init_pool()
    
    def _init_pool(self):
        """Initialize connection pool and batch processor."""
        self._ensure_db_directory()
        self.connection_pool = ConnectionPool(self.db_path, self.max_connections)
        self.batch_processor = BatchProcessor(self.connection_pool)
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool instead of single connection."""
        if not self.connection_pool:
            self._init_pool()
        
        with self.connection_pool.get_connection() as conn:
            yield conn
    
    def initialize(self):
        """Initialize database with optimizations for parallel access."""
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
            
            # Create optimized indexes for parallel queries
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash)',
                'CREATE INDEX IF NOT EXISTS idx_files_size ON files(size)',
                'CREATE INDEX IF NOT EXISTS idx_files_quick_hash ON files(quick_hash)',
                'CREATE INDEX IF NOT EXISTS idx_files_parent ON files(parent_folder_id)',
                'CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_path)',
                'CREATE INDEX IF NOT EXISTS idx_files_size_hash ON files(size, hash)',
                'CREATE INDEX IF NOT EXISTS idx_files_hash_size ON files(hash, size)'
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            # Optimize SQLite for parallel access
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA synchronous=NORMAL')
            cursor.execute('PRAGMA cache_size=10000')
            cursor.execute('PRAGMA temp_store=MEMORY')
            
            conn.commit()
            logger.info("Parallel database initialized successfully")
    
    def batch_insert_files_parallel(self, files: List[FileInfo], 
                                   progress_callback: Optional[Callable] = None):
        """Insert files using parallel batch processing."""
        self.batch_processor.batch_insert_files(files, progress_callback)
    
    def bulk_update_hashes_parallel(self, file_hash_map: Dict[int, str],
                                   progress_callback: Optional[Callable] = None):
        """Update multiple file hashes in parallel."""
        hash_updates = list(file_hash_map.items())
        self.batch_processor.batch_update_hashes(hash_updates, progress_callback)
    
    def get_duplicate_analysis_parallel(self, max_workers: int = 4) -> Tuple[List[Tuple[int, int]], List[Tuple[str, int, int]]]:
        """
        Get duplicate size and hash analysis in parallel.
        
        Args:
            max_workers: Maximum parallel workers
            
        Returns:
            Tuple of (duplicate_sizes, duplicate_hashes)
        """
        queries = [
            '''
            SELECT size, COUNT(*) as count
            FROM files
            GROUP BY size
            HAVING count > 1
            ORDER BY size
            ''',
            '''
            SELECT hash, size, COUNT(*) as count
            FROM files
            WHERE hash IS NOT NULL AND hash != ''
            GROUP BY hash
            HAVING count > 1
            ORDER BY size DESC, hash
            '''
        ]
        
        results = self.batch_processor.parallel_query_processing(queries, max_workers)
        
        duplicate_sizes = [(row['size'], row['count']) for row in results[0]]
        duplicate_hashes = [(row['hash'], row['size'], row['count']) for row in results[1]]
        
        return duplicate_sizes, duplicate_hashes
    
    def calculate_folder_stats_parallel(self, max_workers: int = 4) -> Dict[str, Dict[str, Any]]:
        """
        Calculate folder statistics in parallel.
        
        Args:
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary of folder statistics
        """
        # Get all folders first
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, path FROM folders')
            folders = [(row['id'], row['path']) for row in cursor.fetchall()]
        
        if not folders:
            return {}
        
        results = {}
        
        def calculate_folder_stats(folder_data):
            folder_id, folder_path = folder_data
            try:
                with self.connection_pool.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Get basic folder stats
                    cursor.execute(
                        'SELECT COUNT(*) as file_count, SUM(size) as total_size FROM files WHERE parent_folder_id = ?',
                        (folder_id,)
                    )
                    basic_stats = cursor.fetchone()
                    
                    # Get duplication stats
                    cursor.execute('''
                        SELECT COUNT(DISTINCT f1.id) as duplicated_count
                        FROM files f1
                        INNER JOIN files f2 ON f1.hash = f2.hash
                        WHERE f1.parent_folder_id = ?
                          AND f2.id != f1.id
                          AND f1.hash IS NOT NULL
                          AND f1.hash != ''
                    ''', (folder_id,))
                    
                    dup_stats = cursor.fetchone()
                    
                    file_count = basic_stats['file_count'] or 0
                    duplicated_count = dup_stats['duplicated_count'] or 0
                    duplication_pct = (duplicated_count / file_count * 100) if file_count > 0 else 0
                    
                    return folder_path, {
                        'file_count': file_count,
                        'total_size': basic_stats['total_size'] or 0,
                        'duplicated_files': duplicated_count,
                        'duplication_percentage': duplication_pct
                    }
            except Exception as e:
                logger.error(f"Error calculating stats for folder {folder_path}: {e}")
                return folder_path, {}
        
        # Process folders in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_folder = {
                executor.submit(calculate_folder_stats, folder_data): folder_data[1]
                for folder_data in folders
            }
            
            with tqdm(total=len(folders), desc="Calculating folder stats", unit='folders') as pbar:
                for future in as_completed(future_to_folder):
                    folder_path, stats = future.result()
                    results[folder_path] = stats
                    pbar.update(1)
        
        return results
    
    def vacuum_and_optimize(self):
        """Perform database maintenance and optimization."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Analyze tables for query optimization
            cursor.execute('ANALYZE')
            
            # Update statistics
            cursor.execute('PRAGMA optimize')
            
            logger.info("Database optimized")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        stats = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Database size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats['database_size_mb'] = (page_count * page_size) / (1024 * 1024)
            
            # Cache stats
            cursor.execute("PRAGMA cache_size")
            stats['cache_size'] = cursor.fetchone()[0]
            
            # WAL mode info
            cursor.execute("PRAGMA journal_mode")
            stats['journal_mode'] = cursor.fetchone()[0]
            
            # Table counts
            cursor.execute("SELECT COUNT(*) FROM folders")
            stats['total_folders'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM files")
            stats['total_files'] = cursor.fetchone()[0]
            
            # Index usage
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            stats['index_count'] = len(cursor.fetchall())
        
        return stats
    
    def close(self):
        """Close all database connections."""
        if self.connection_pool:
            self.connection_pool.close_all()
            self.connection_pool = None
        
        if self.connection:
            self.connection.close()
            self.connection = None
        
        logger.info("All database connections closed")