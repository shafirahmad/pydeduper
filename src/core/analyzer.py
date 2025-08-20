"""Duplicate detection and analysis functionality for PyDeduper."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

from ..config import Config
from ..storage.models import (
    FileInfo, FolderInfo, DuplicateGroup, 
    AnalysisResult, ScanResult
)
from ..storage.database import Database
from .hasher import FileHasher, HashCache
from .scanner import FileScanner

logger = logging.getLogger(__name__)


class DuplicateAnalyzer:
    """Analyzes files to find duplicates and calculate statistics."""
    
    def __init__(self, config: Config = None):
        """Initialize the analyzer."""
        self.config = config or Config()
        self.hasher = FileHasher(config)
        self.cache = HashCache()
        self.db = Database(config)
    
    def find_duplicates_by_size(self, files: List[FileInfo]) -> Dict[int, List[FileInfo]]:
        """
        Group files by size to find potential duplicates.
        
        Args:
            files: List of FileInfo objects
            
        Returns:
            Dictionary mapping size to list of files with that size
        """
        size_groups = defaultdict(list)
        
        for file in files:
            size_groups[file.size].append(file)
        
        # Filter out sizes with only one file
        return {
            size: file_list 
            for size, file_list in size_groups.items() 
            if len(file_list) > 1
        }
    
    def calculate_hashes_for_duplicates(self, 
                                       size_groups: Dict[int, List[FileInfo]], 
                                       use_quick_hash: bool = True) -> Dict[str, List[FileInfo]]:
        """
        Calculate hashes for files that have duplicate sizes.
        
        Args:
            size_groups: Dictionary of size to file lists
            use_quick_hash: Whether to use quick hash first
            
        Returns:
            Dictionary mapping hash to list of files with that hash
        """
        hash_groups = defaultdict(list)
        
        for size, files in size_groups.items():
            if use_quick_hash and size > self.config.QUICK_HASH_SIZE * 3:
                # First pass with quick hash for large files
                quick_hash_groups = defaultdict(list)
                
                for file in files:
                    # Check cache first
                    mtime = file.last_modified.timestamp() if file.last_modified else None
                    quick_hash = self.cache.get_quick_hash(file.path, mtime)
                    
                    if not quick_hash:
                        quick_hash = self.hasher.quick_hash(file.path)
                        if quick_hash:
                            self.cache.set_quick_hash(file.path, quick_hash, mtime)
                    
                    if quick_hash:
                        file.quick_hash = quick_hash
                        quick_hash_groups[quick_hash].append(file)
                
                # Only calculate full hash for files with duplicate quick hashes
                for quick_hash, quick_files in quick_hash_groups.items():
                    if len(quick_files) > 1:
                        for file in quick_files:
                            full_hash = self._calculate_full_hash(file)
                            if full_hash:
                                hash_groups[full_hash].append(file)
            else:
                # Direct full hash for small files
                for file in files:
                    full_hash = self._calculate_full_hash(file)
                    if full_hash:
                        hash_groups[full_hash].append(file)
        
        # Filter out hashes with only one file
        return {
            hash_val: file_list 
            for hash_val, file_list in hash_groups.items() 
            if len(file_list) > 1
        }
    
    def _calculate_full_hash(self, file: FileInfo) -> Optional[str]:
        """Calculate full hash for a file, using cache if available."""
        mtime = file.last_modified.timestamp() if file.last_modified else None
        
        # Check cache first
        hash_value = self.cache.get_hash(file.path, mtime)
        
        if not hash_value:
            hash_value = self.hasher.hash_file(file.path)
            if hash_value:
                self.cache.set_hash(file.path, hash_value, mtime)
        
        if hash_value:
            file.hash = hash_value
        
        return hash_value
    
    def analyze_folder(self, folder: FolderInfo) -> AnalysisResult:
        """
        Analyze a folder for duplicates.
        
        Args:
            folder: FolderInfo object to analyze
            
        Returns:
            AnalysisResult with duplicate information
        """
        result = AnalysisResult()
        
        # Collect all files recursively
        all_files = self._collect_files_recursive(folder)
        
        # Find duplicates by size
        size_groups = self.find_duplicates_by_size(all_files)
        
        if not size_groups:
            return result
        
        # Calculate hashes for potential duplicates
        hash_groups = self.calculate_hashes_for_duplicates(size_groups)
        
        # Create duplicate groups
        for hash_val, files in hash_groups.items():
            if len(files) > 1:
                group = DuplicateGroup(
                    hash=hash_val,
                    size=files[0].size,
                    count=len(files),
                    files=files
                )
                result.duplicate_groups.append(group)
        
        # Calculate statistics
        result.calculate_stats()
        
        # Calculate folder duplication percentages
        result.folder_duplication_stats = self._calculate_folder_duplication_stats(folder, hash_groups)
        
        return result
    
    def _collect_files_recursive(self, folder: FolderInfo) -> List[FileInfo]:
        """Recursively collect all files from a folder and its subfolders."""
        all_files = list(folder.files)
        
        for subfolder in folder.subfolders:
            all_files.extend(self._collect_files_recursive(subfolder))
        
        return all_files
    
    def _calculate_folder_duplication_stats(self, 
                                           root_folder: FolderInfo, 
                                           hash_groups: Dict[str, List[FileInfo]]) -> Dict[str, float]:
        """
        Calculate duplication percentage for each folder.
        
        Returns:
            Dictionary mapping folder path to duplication percentage
        """
        stats = {}
        
        # Build a set of duplicate hashes
        duplicate_hashes = set(hash_groups.keys())
        
        # Calculate for each folder
        def calculate_for_folder(folder: FolderInfo):
            if not folder.files:
                stats[folder.full_path] = 0.0
            else:
                duplicated_count = sum(
                    1 for file in folder.files 
                    if file.hash and file.hash in duplicate_hashes
                )
                stats[folder.full_path] = (duplicated_count / len(folder.files)) * 100
            
            # Recurse into subfolders
            for subfolder in folder.subfolders:
                calculate_for_folder(subfolder)
        
        calculate_for_folder(root_folder)
        return stats
    
    def analyze_with_database(self, paths: List[Path]) -> AnalysisResult:
        """
        Analyze files using database for persistence.
        
        Args:
            paths: List of paths to analyze
            
        Returns:
            AnalysisResult with duplicate information
        """
        # Initialize database
        self.db.initialize()
        
        # Clear existing data (optional, could be configurable)
        if self.config.VERBOSE:
            logger.info("Clearing existing database...")
        self.db.clear_all()
        
        # Scan files
        scanner = FileScanner(self.config)
        folders = scanner.scan_multiple(paths)
        
        # Store in database
        for folder in folders:
            self._store_folder_recursive(folder)
        
        # Find duplicate sizes
        duplicate_sizes = self.db.get_duplicate_sizes()
        
        # Calculate hashes for files with duplicate sizes
        for size, count in duplicate_sizes:
            files_data = self.db.get_files_by_size(size)
            
            for file_data in files_data:
                if not file_data['hash']:
                    file_path = Path(file_data['path'])
                    hash_value = self.hasher.hash_file(file_path)
                    if hash_value:
                        self.db.update_file_hash(file_data['id'], hash_value)
        
        # Get duplicate hashes and build result
        result = AnalysisResult()
        duplicate_hashes = self.db.get_duplicate_hashes()
        
        for hash_val, size, count in duplicate_hashes:
            files_data = self.db.get_files_by_hash(hash_val)
            files = [FileInfo.from_dict(f) for f in files_data]
            
            group = DuplicateGroup(
                hash=hash_val,
                size=size,
                count=count,
                files=files
            )
            result.duplicate_groups.append(group)
        
        result.calculate_stats()
        
        # Calculate folder duplication stats from database
        result.folder_duplication_stats = self._calculate_db_folder_stats()
        
        return result
    
    def _store_folder_recursive(self, folder: FolderInfo):
        """Recursively store folder and its contents in the database."""
        # Store folder
        folder_id = self.db.insert_folder(folder)
        
        # Store files
        if folder.files:
            for file in folder.files:
                file.parent_folder_id = folder_id
            self.db.batch_insert_files(folder.files, folder_id)
        
        # Recurse into subfolders
        for subfolder in folder.subfolders:
            self._store_folder_recursive(subfolder)
    
    def _calculate_db_folder_stats(self) -> Dict[str, float]:
        """Calculate folder duplication statistics from database."""
        stats = {}
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, path FROM folders')
            
            for row in cursor.fetchall():
                folder_id = row['id']
                folder_path = row['path']
                duplication_pct = self.db.calculate_folder_duplication(folder_id)
                stats[folder_path] = duplication_pct
        
        return stats
    
    def find_similar_folders(self, threshold: float = 70.0) -> List[Tuple[str, str, float]]:
        """
        Find folders with similar content based on file hashes.
        
        Args:
            threshold: Minimum similarity percentage
            
        Returns:
            List of (folder1, folder2, similarity_percentage) tuples
        """
        similar_pairs = []
        
        # Get all folders with their file hashes
        folder_hashes = self._get_folder_hashes()
        
        # Compare each pair of folders
        folder_paths = list(folder_hashes.keys())
        for i in range(len(folder_paths)):
            for j in range(i + 1, len(folder_paths)):
                path1, path2 = folder_paths[i], folder_paths[j]
                hashes1, hashes2 = folder_hashes[path1], folder_hashes[path2]
                
                if hashes1 and hashes2:
                    similarity = self._calculate_similarity(hashes1, hashes2)
                    if similarity >= threshold:
                        similar_pairs.append((path1, path2, similarity))
        
        return sorted(similar_pairs, key=lambda x: x[2], reverse=True)
    
    def _get_folder_hashes(self) -> Dict[str, Set[str]]:
        """Get a mapping of folder paths to sets of file hashes."""
        folder_hashes = {}
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.path as folder_path, fi.hash
                FROM folders f
                JOIN files fi ON f.id = fi.parent_folder_id
                WHERE fi.hash IS NOT NULL AND fi.hash != ''
            ''')
            
            for row in cursor.fetchall():
                folder_path = row['folder_path']
                file_hash = row['hash']
                
                if folder_path not in folder_hashes:
                    folder_hashes[folder_path] = set()
                folder_hashes[folder_path].add(file_hash)
        
        return folder_hashes
    
    def _calculate_similarity(self, hashes1: Set[str], hashes2: Set[str]) -> float:
        """Calculate similarity percentage between two sets of hashes."""
        if not hashes1 or not hashes2:
            return 0.0
        
        intersection = hashes1 & hashes2
        union = hashes1 | hashes2
        
        return (len(intersection) / len(union)) * 100 if union else 0.0