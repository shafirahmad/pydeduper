"""Tests for parallel processing components."""

import unittest
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

from src.core.parallel_scanner import ParallelScanner, ProgressReporter
from src.core.parallel_hasher import ParallelHasher, HashTask, AdaptiveHasher
from src.core.parallel_analyzer import ProgressTracker, ParallelDuplicateAnalyzer
from src.storage.parallel_database import ConnectionPool, BatchProcessor, ParallelDatabase
from src.storage.models import FileInfo
from src.config import Config


class TestProgressReporter(unittest.TestCase):
    """Test cases for ProgressReporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reporter = ProgressReporter(enable_progress=False)  # Disable actual progress bars
    
    def test_create_progress_bar(self):
        """Test progress bar creation."""
        pbar = self.reporter.create_progress_bar("test", 100, "Test progress")
        self.assertIsNone(pbar)  # Should be None when disabled
    
    def test_update_progress(self):
        """Test progress updates."""
        self.reporter.create_progress_bar("test", 100)
        # Should not raise exception even when progress bar doesn't exist
        self.reporter.update_progress("test", 1, "Updated")
    
    def test_cancellation(self):
        """Test cancellation functionality."""
        self.assertFalse(self.reporter.is_cancelled())
        self.reporter.cancel()
        self.assertTrue(self.reporter.is_cancelled())


class TestParallelScanner(unittest.TestCase):
    """Test cases for ParallelScanner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.scanner = ParallelScanner(self.config, max_workers=2)
        self.scanner.set_progress_enabled(False)  # Disable progress for tests
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self):
        """Create test file structure."""
        test_files = [
            ("file1.txt", b"Content 1"),
            ("file2.txt", b"Content 2"),
            ("subdir/file3.txt", b"Content 3"),
        ]
        
        for path, content in test_files:
            file_path = Path(self.temp_dir) / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
        
        return test_files
    
    def test_count_total_items(self):
        """Test counting total items for progress."""
        self.create_test_files()
        total = self.scanner.count_total_items([Path(self.temp_dir)])
        self.assertGreater(total, 0)
    
    def test_scan_tree_parallel(self):
        """Test parallel tree scanning."""
        self.create_test_files()
        root = self.scanner.scan_tree_parallel(Path(self.temp_dir))
        
        self.assertIsNotNone(root)
        self.assertEqual(len(root.files), 2)  # file1.txt, file2.txt
        self.assertEqual(len(root.subfolders), 1)  # subdir
    
    def test_cancellation(self):
        """Test scan cancellation."""
        self.scanner.cancel()
        self.assertTrue(self.scanner.is_cancelled())


class TestParallelHasher(unittest.TestCase):
    """Test cases for ParallelHasher."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.hasher = ParallelHasher(self.config, max_workers=2)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self):
        """Create test files for hashing."""
        files = []
        contents = [b"Content A", b"Content B", b"Content A"]  # A duplicate
        
        for i, content in enumerate(contents):
            file_path = Path(self.temp_dir) / f"file{i}.txt"
            file_path.write_bytes(content)
            
            file_info = FileInfo(
                path=file_path,
                name=file_path.name,
                size=len(content)
            )
            files.append(file_info)
        
        return files
    
    def test_calculate_hashes_parallel(self):
        """Test parallel hash calculation."""
        files = self.create_test_files()
        
        results = self.hasher.calculate_hashes_parallel(files, 'full')
        
        self.assertEqual(len(results), 3)
        # First and third files should have same hash (same content)
        hash1 = results[str(files[0].path)]
        hash3 = results[str(files[2].path)]
        self.assertEqual(hash1, hash3)
    
    def test_hash_caching(self):
        """Test hash caching functionality."""
        files = self.create_test_files()
        
        # First calculation
        results1 = self.hasher.calculate_hashes_parallel(files, 'full')
        
        # Second calculation should use cache
        results2 = self.hasher.calculate_hashes_parallel(files, 'full')
        
        self.assertEqual(results1, results2)
        self.assertGreater(self.hasher.stats['cache_hits'], 0)
    
    def test_two_stage_hashing(self):
        """Test two-stage hashing for large files."""
        # Create larger files to trigger two-stage hashing
        files = []
        for i in range(2):
            content = b"A" * (self.config.QUICK_HASH_SIZE * 5)  # Large file
            file_path = Path(self.temp_dir) / f"large{i}.txt"
            file_path.write_bytes(content)
            
            file_info = FileInfo(
                path=file_path,
                name=file_path.name,
                size=len(content)
            )
            files.append(file_info)
        
        results = self.hasher._two_stage_hashing(files)
        self.assertEqual(len(results), 2)
    
    def test_statistics(self):
        """Test statistics collection."""
        files = self.create_test_files()
        self.hasher.calculate_hashes_parallel(files, 'full')
        
        stats = self.hasher.get_statistics()
        self.assertIn('files_processed', stats)
        self.assertGreater(stats['files_processed'], 0)


class TestAdaptiveHasher(unittest.TestCase):
    """Test cases for AdaptiveHasher."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.hasher = AdaptiveHasher(self.config, max_workers=2)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_strategy_selection_small_files(self):
        """Test strategy selection for small files."""
        files = [
            FileInfo(path=Path("small1.txt"), name="small1.txt", size=100),
            FileInfo(path=Path("small2.txt"), name="small2.txt", size=200),
        ]
        
        strategy = self.hasher.choose_hash_strategy(files)
        self.assertEqual(strategy, 'full')
    
    def test_strategy_selection_large_files(self):
        """Test strategy selection for very large files."""
        files = [
            FileInfo(path=Path("large1.txt"), name="large1.txt", size=200 * 1024 * 1024),  # 200MB
        ]
        
        strategy = self.hasher.choose_hash_strategy(files)
        self.assertEqual(strategy, 'partial')
    
    def test_strategy_selection_medium_files(self):
        """Test strategy selection for medium files."""
        files = [
            FileInfo(path=Path("med1.txt"), name="med1.txt", size=50 * 1024),  # 50KB
        ]
        
        strategy = self.hasher.choose_hash_strategy(files)
        self.assertEqual(strategy, 'quick')


class TestProgressTracker(unittest.TestCase):
    """Test cases for ProgressTracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock tqdm to avoid actual progress bars in tests
        self.patcher = patch('src.core.parallel_analyzer.tqdm')
        self.mock_tqdm = self.patcher.start()
        self.mock_tqdm.return_value.__enter__ = MagicMock(return_value=self.mock_tqdm.return_value)
        self.mock_tqdm.return_value.__exit__ = MagicMock(return_value=None)
        
        self.tracker = ProgressTracker(3, ["Phase 1", "Phase 2", "Phase 3"])
    
    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()
    
    def test_phase_management(self):
        """Test phase start/finish cycle."""
        self.tracker.start_phase(0, 100, "Test phase")
        self.assertEqual(self.tracker.current_phase, 0)
        
        self.tracker.update_phase(10, "Processing...")
        self.assertEqual(self.tracker.phase_progress[0]['completed'], 10)
        
        self.tracker.finish_phase()
        # Main progress should be updated
    
    def test_cancellation(self):
        """Test cancellation functionality."""
        self.assertFalse(self.tracker.is_cancelled())
        self.tracker.cancel()
        self.assertTrue(self.tracker.is_cancelled())


class TestConnectionPool(unittest.TestCase):
    """Test cases for ConnectionPool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = Path(tempfile.mktemp(suffix='.db'))
        self.pool = ConnectionPool(self.temp_db, max_connections=3)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.pool.close_all()
        if self.temp_db.exists():
            self.temp_db.unlink()
    
    def test_connection_pool_basic(self):
        """Test basic connection pool functionality."""
        with self.pool.get_connection() as conn:
            self.assertIsNotNone(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)
    
    def test_concurrent_connections(self):
        """Test concurrent connection usage."""
        def use_connection():
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return cursor.fetchone()[0]
        
        # Test multiple concurrent connections
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(use_connection) for _ in range(5)]
            results = [future.result() for future in futures]
        
        self.assertEqual(results, [1] * 5)


class TestBatchProcessor(unittest.TestCase):
    """Test cases for BatchProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = Path(tempfile.mktemp(suffix='.db'))
        self.pool = ConnectionPool(self.temp_db, max_connections=2)
        self.processor = BatchProcessor(self.pool, batch_size=2)
        
        # Initialize database
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    hash TEXT,
                    quick_hash TEXT,
                    parent_folder_id INTEGER,
                    last_modified TIMESTAMP
                )
            ''')
            conn.commit()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.pool.close_all()
        if self.temp_db.exists():
            self.temp_db.unlink()
    
    def test_batch_insert_files(self):
        """Test batch file insertion."""
        files = [
            FileInfo(path=Path(f"/test/file{i}.txt"), name=f"file{i}.txt", size=100 + i)
            for i in range(5)
        ]
        
        self.processor.batch_insert_files(files)
        
        # Verify insertion
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM files")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 5)


if __name__ == "__main__":
    unittest.main()