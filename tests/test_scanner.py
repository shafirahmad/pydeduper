"""Tests for the scanner module."""

import unittest
import tempfile
import os
from pathlib import Path
from datetime import datetime

from src.core.scanner import FileScanner
from src.config import Config


class TestFileScanner(unittest.TestCase):
    """Test cases for FileScanner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.scanner = FileScanner(self.config)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_structure(self):
        """Create a test directory structure."""
        # Create directories
        (Path(self.temp_dir) / "subdir1").mkdir()
        (Path(self.temp_dir) / "subdir2").mkdir()
        (Path(self.temp_dir) / "subdir1" / "nested").mkdir()
        
        # Create files
        files = [
            ("file1.txt", b"Content 1"),
            ("file2.txt", b"Content 2"),
            ("subdir1/file3.txt", b"Content 3"),
            ("subdir1/nested/file4.txt", b"Content 4"),
            ("subdir2/file5.txt", b"Content 5"),
        ]
        
        for path, content in files:
            file_path = Path(self.temp_dir) / path
            file_path.write_bytes(content)
        
        return files
    
    def test_scan_single_file(self):
        """Test scanning a single file."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = b"Test content"
        test_file.write_bytes(test_content)
        
        file_info = self.scanner.scan_file(test_file)
        
        self.assertIsNotNone(file_info)
        self.assertEqual(file_info.name, "test.txt")
        self.assertEqual(file_info.size, len(test_content))
        self.assertEqual(file_info.path, test_file)
    
    def test_scan_nonexistent_file(self):
        """Test scanning a non-existent file returns None."""
        fake_file = Path(self.temp_dir) / "nonexistent.txt"
        file_info = self.scanner.scan_file(fake_file)
        
        self.assertIsNone(file_info)
        self.assertTrue(self.scanner.scan_result.has_errors)
    
    def test_scan_directory_non_recursive(self):
        """Test non-recursive directory scanning."""
        self.create_test_structure()
        
        results = list(self.scanner.scan_directory(Path(self.temp_dir), recursive=False))
        
        # Should only get the root directory
        self.assertEqual(len(results), 1)
        folder_path, files = results[0]
        self.assertEqual(folder_path, Path(self.temp_dir))
        self.assertEqual(len(files), 2)  # Only file1.txt and file2.txt
    
    def test_scan_directory_recursive(self):
        """Test recursive directory scanning."""
        self.create_test_structure()
        
        results = list(self.scanner.scan_directory(Path(self.temp_dir), recursive=True))
        
        # Should get root + 3 subdirectories
        self.assertEqual(len(results), 4)
        
        # Check total files found
        total_files = sum(len(files) for _, files in results)
        self.assertEqual(total_files, 5)
    
    def test_scan_tree(self):
        """Test building complete directory tree."""
        self.create_test_structure()
        
        root = self.scanner.scan_tree(Path(self.temp_dir))
        
        self.assertEqual(root.name, os.path.basename(self.temp_dir))
        self.assertEqual(len(root.files), 2)  # Root files
        self.assertEqual(len(root.subfolders), 2)  # subdir1 and subdir2
        
        # Check nested structure
        subdir1 = next((f for f in root.subfolders if f.name == "subdir1"), None)
        self.assertIsNotNone(subdir1)
        self.assertEqual(len(subdir1.files), 1)  # file3.txt
        self.assertEqual(len(subdir1.subfolders), 1)  # nested
    
    def test_ignore_patterns(self):
        """Test that ignore patterns work correctly."""
        # Create files that should be ignored
        (Path(self.temp_dir) / "file.txt").write_bytes(b"Normal")
        (Path(self.temp_dir) / "file.tmp").write_bytes(b"Temp")
        (Path(self.temp_dir) / "~backup").write_bytes(b"Backup")
        
        results = list(self.scanner.scan_directory(Path(self.temp_dir)))
        folder_path, files = results[0]
        
        # Should only find file.txt
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "file.txt")
    
    def test_ignore_hidden_files(self):
        """Test ignoring hidden files when configured."""
        # Create hidden and normal files
        (Path(self.temp_dir) / "normal.txt").write_bytes(b"Normal")
        (Path(self.temp_dir) / ".hidden").write_bytes(b"Hidden")
        
        # Test with hidden files ignored
        self.config.IGNORE_HIDDEN = True
        scanner = FileScanner(self.config)
        
        results = list(scanner.scan_directory(Path(self.temp_dir)))
        folder_path, files = results[0]
        
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "normal.txt")
        
        # Test with hidden files included
        self.config.IGNORE_HIDDEN = False
        scanner = FileScanner(self.config)
        
        results = list(scanner.scan_directory(Path(self.temp_dir)))
        folder_path, files = results[0]
        
        self.assertEqual(len(files), 2)
    
    def test_scan_result_statistics(self):
        """Test that scan results are properly tracked."""
        self.create_test_structure()
        
        self.scanner.scan_tree(Path(self.temp_dir))
        result = self.scanner.get_scan_result()
        
        self.assertEqual(result.total_files, 5)
        self.assertEqual(result.total_folders, 4)
        self.assertGreater(result.total_size, 0)
        self.assertGreater(result.scan_duration, 0)
        self.assertFalse(result.has_errors)


if __name__ == "__main__":
    unittest.main()