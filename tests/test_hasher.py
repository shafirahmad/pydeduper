"""Tests for the hasher module."""

import unittest
import tempfile
import hashlib
from pathlib import Path

from src.core.hasher import FileHasher, HashCache
from src.config import Config, HashAlgorithm


class TestFileHasher(unittest.TestCase):
    """Test cases for FileHasher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.hasher = FileHasher(self.config)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, content: bytes, name: str = "test.txt") -> Path:
        """Create a test file with given content."""
        file_path = Path(self.temp_dir) / name
        file_path.write_bytes(content)
        return file_path
    
    def test_hash_file_sha256(self):
        """Test SHA256 hashing of a file."""
        content = b"Hello, World!"
        file_path = self.create_test_file(content)
        
        self.hasher.algorithm = HashAlgorithm.SHA256
        result = self.hasher.hash_file(file_path)
        
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(result, expected)
    
    def test_hash_file_md5(self):
        """Test MD5 hashing of a file."""
        content = b"Test content for MD5"
        file_path = self.create_test_file(content)
        
        self.hasher.algorithm = HashAlgorithm.MD5
        result = self.hasher.hash_file(file_path)
        
        expected = hashlib.md5(content).hexdigest()
        self.assertEqual(result, expected)
    
    def test_quick_hash(self):
        """Test quick hash functionality."""
        content = b"A" * 2000  # Content larger than quick hash size
        file_path = self.create_test_file(content)
        
        result = self.hasher.quick_hash(file_path)
        
        # Quick hash should only use first QUICK_HASH_SIZE bytes
        expected_content = content[:self.config.QUICK_HASH_SIZE]
        expected = hashlib.sha256(expected_content).hexdigest()
        self.assertEqual(result, expected)
    
    def test_hash_nonexistent_file(self):
        """Test hashing a non-existent file returns None."""
        fake_path = Path(self.temp_dir) / "nonexistent.txt"
        result = self.hasher.hash_file(fake_path)
        self.assertIsNone(result)
    
    def test_verify_hash(self):
        """Test hash verification."""
        content = b"Verification test"
        file_path = self.create_test_file(content)
        
        hash_value = self.hasher.hash_file(file_path)
        self.assertTrue(self.hasher.verify_hash(file_path, hash_value))
        self.assertFalse(self.hasher.verify_hash(file_path, "wronghash"))
    
    def test_identical_files_same_hash(self):
        """Test that identical files produce the same hash."""
        content = b"Duplicate content"
        file1 = self.create_test_file(content, "file1.txt")
        file2 = self.create_test_file(content, "file2.txt")
        
        hash1 = self.hasher.hash_file(file1)
        hash2 = self.hasher.hash_file(file2)
        
        self.assertEqual(hash1, hash2)
    
    def test_different_files_different_hash(self):
        """Test that different files produce different hashes."""
        file1 = self.create_test_file(b"Content 1", "file1.txt")
        file2 = self.create_test_file(b"Content 2", "file2.txt")
        
        hash1 = self.hasher.hash_file(file1)
        hash2 = self.hasher.hash_file(file2)
        
        self.assertNotEqual(hash1, hash2)


class TestHashCache(unittest.TestCase):
    """Test cases for HashCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = HashCache()
        self.test_path = Path("/test/file.txt")
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = self.cache.get_hash(self.test_path)
        self.assertIsNone(result)
    
    def test_cache_hit(self):
        """Test cache hit returns stored value."""
        hash_value = "testhash123"
        mtime = 123456.789
        
        self.cache.set_hash(self.test_path, hash_value, mtime)
        result = self.cache.get_hash(self.test_path, mtime)
        
        self.assertEqual(result, hash_value)
    
    def test_cache_invalidation_on_mtime_change(self):
        """Test cache returns None when mtime changes."""
        hash_value = "testhash123"
        old_mtime = 123456.789
        new_mtime = 123457.789
        
        self.cache.set_hash(self.test_path, hash_value, old_mtime)
        result = self.cache.get_hash(self.test_path, new_mtime)
        
        self.assertIsNone(result)
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        self.cache.set_hash(self.test_path, "hash1", 123)
        self.cache.set_quick_hash(self.test_path, "hash2", 123)
        
        self.cache.clear()
        
        self.assertIsNone(self.cache.get_hash(self.test_path, 123))
        self.assertIsNone(self.cache.get_quick_hash(self.test_path, 123))
    
    def test_remove_specific_file(self):
        """Test removing a specific file from cache."""
        path1 = Path("/test/file1.txt")
        path2 = Path("/test/file2.txt")
        
        self.cache.set_hash(path1, "hash1", 123)
        self.cache.set_hash(path2, "hash2", 123)
        
        self.cache.remove(path1)
        
        self.assertIsNone(self.cache.get_hash(path1, 123))
        self.assertEqual(self.cache.get_hash(path2, 123), "hash2")


if __name__ == "__main__":
    unittest.main()