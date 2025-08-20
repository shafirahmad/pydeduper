"""Tests for the analyzer module."""

import unittest
import tempfile
from pathlib import Path
from typing import List

from src.core.analyzer import DuplicateAnalyzer
from src.storage.models import FileInfo, FolderInfo
from src.config import Config


class TestDuplicateAnalyzer(unittest.TestCase):
    """Test cases for DuplicateAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.analyzer = DuplicateAnalyzer(self.config)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.analyzer.db.close()
    
    def create_test_files(self) -> List[FileInfo]:
        """Create test FileInfo objects."""
        files = [
            FileInfo(path=Path("/test/file1.txt"), name="file1.txt", size=100),
            FileInfo(path=Path("/test/file2.txt"), name="file2.txt", size=100),
            FileInfo(path=Path("/test/file3.txt"), name="file3.txt", size=200),
            FileInfo(path=Path("/test/file4.txt"), name="file4.txt", size=100),
            FileInfo(path=Path("/test/file5.txt"), name="file5.txt", size=300),
        ]
        return files
    
    def test_find_duplicates_by_size(self):
        """Test grouping files by size."""
        files = self.create_test_files()
        
        size_groups = self.analyzer.find_duplicates_by_size(files)
        
        # Should find one group with size 100 (3 files)
        self.assertEqual(len(size_groups), 1)
        self.assertIn(100, size_groups)
        self.assertEqual(len(size_groups[100]), 3)
    
    def test_no_duplicates_by_size(self):
        """Test when no files have duplicate sizes."""
        files = [
            FileInfo(path=Path("/test/file1.txt"), name="file1.txt", size=100),
            FileInfo(path=Path("/test/file2.txt"), name="file2.txt", size=200),
            FileInfo(path=Path("/test/file3.txt"), name="file3.txt", size=300),
        ]
        
        size_groups = self.analyzer.find_duplicates_by_size(files)
        
        self.assertEqual(len(size_groups), 0)
    
    def test_analyze_folder_no_duplicates(self):
        """Test analyzing folder with no duplicates."""
        folder = FolderInfo(
            path=Path("/test"),
            name="test",
            files=[
                FileInfo(path=Path("/test/file1.txt"), name="file1.txt", size=100),
                FileInfo(path=Path("/test/file2.txt"), name="file2.txt", size=200),
                FileInfo(path=Path("/test/file3.txt"), name="file3.txt", size=300),
            ]
        )
        
        result = self.analyzer.analyze_folder(folder)
        
        self.assertEqual(len(result.duplicate_groups), 0)
        self.assertEqual(result.total_duplicates, 0)
        self.assertEqual(result.total_wasted_space, 0)
    
    def test_analyze_folder_with_duplicates(self):
        """Test analyzing folder with duplicate files."""
        # Create actual files for hashing
        file1_path = Path(self.temp_dir) / "file1.txt"
        file2_path = Path(self.temp_dir) / "file2.txt"
        file3_path = Path(self.temp_dir) / "file3.txt"
        
        # Create files with same content (duplicates)
        duplicate_content = b"Duplicate content"
        file1_path.write_bytes(duplicate_content)
        file2_path.write_bytes(duplicate_content)
        file3_path.write_bytes(b"Different content")
        
        folder = FolderInfo(
            path=Path(self.temp_dir),
            name="test",
            files=[
                FileInfo(path=file1_path, name="file1.txt", size=len(duplicate_content)),
                FileInfo(path=file2_path, name="file2.txt", size=len(duplicate_content)),
                FileInfo(path=file3_path, name="file3.txt", size=17),
            ]
        )
        
        result = self.analyzer.analyze_folder(folder)
        
        # Should find 1 duplicate group with 2 files
        self.assertEqual(len(result.duplicate_groups), 1)
        self.assertEqual(result.duplicate_groups[0].count, 2)
        self.assertEqual(result.total_duplicates, 1)
        self.assertEqual(result.total_wasted_space, len(duplicate_content))
    
    def test_folder_duplication_percentage(self):
        """Test calculation of folder duplication percentage."""
        # Create test files
        file1_path = Path(self.temp_dir) / "file1.txt"
        file2_path = Path(self.temp_dir) / "file2.txt"
        
        duplicate_content = b"Duplicate"
        file1_path.write_bytes(duplicate_content)
        file2_path.write_bytes(duplicate_content)
        
        folder = FolderInfo(
            path=Path(self.temp_dir),
            name="test",
            files=[
                FileInfo(path=file1_path, name="file1.txt", size=len(duplicate_content)),
                FileInfo(path=file2_path, name="file2.txt", size=len(duplicate_content)),
            ]
        )
        
        result = self.analyzer.analyze_folder(folder)
        
        # Both files are duplicates, so 100% duplication
        folder_path = str(folder.path)
        self.assertIn(folder_path, result.folder_duplication_stats)
        self.assertEqual(result.folder_duplication_stats[folder_path], 100.0)
    
    def test_database_operations(self):
        """Test database storage and retrieval."""
        # Initialize database
        self.analyzer.db.initialize()
        
        # Create and store a folder
        folder = FolderInfo(
            path=Path("/test/folder"),
            name="folder",
            num_files=2,
            total_size=200
        )
        
        folder_id = self.analyzer.db.insert_folder(folder)
        self.assertIsNotNone(folder_id)
        
        # Create and store files
        file1 = FileInfo(
            path=Path("/test/folder/file1.txt"),
            name="file1.txt",
            size=100,
            hash="hash123"
        )
        
        file_id = self.analyzer.db.insert_file(file1, folder_id)
        self.assertIsNotNone(file_id)
        
        # Retrieve folder
        folder_data = self.analyzer.db.get_folder_by_path(Path("/test/folder"))
        self.assertIsNotNone(folder_data)
        self.assertEqual(folder_data['name'], "folder")
        
        # Retrieve files
        files = self.analyzer.db.get_files_by_folder(folder_id)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]['name'], "file1.txt")


if __name__ == "__main__":
    unittest.main()