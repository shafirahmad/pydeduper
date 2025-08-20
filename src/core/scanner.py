"""File system scanning functionality for PyDeduper."""

import os
import time
import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, List
from datetime import datetime
import fnmatch

from ..config import Config
from ..storage.models import FileInfo, FolderInfo, ScanResult

logger = logging.getLogger(__name__)


class FileScanner:
    """Handles file system traversal and scanning operations."""
    
    def __init__(self, config: Config = None):
        """Initialize the scanner with configuration."""
        self.config = config or Config()
        self.scan_result = ScanResult()
    
    def should_ignore(self, path: Path) -> bool:
        """
        Check if a file or folder should be ignored based on patterns.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path should be ignored
        """
        name = path.name
        
        # Check if hidden and we're ignoring hidden files
        if self.config.IGNORE_HIDDEN and name.startswith('.'):
            return True
        
        # Check against ignore patterns
        for pattern in self.config.IGNORE_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                return True
        
        return False
    
    def scan_file(self, file_path: Path) -> Optional[FileInfo]:
        """
        Scan a single file and return its information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileInfo object or None if error
        """
        try:
            stat = file_path.stat()
            
            file_info = FileInfo(
                path=file_path,
                name=file_path.name,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime)
            )
            
            return file_info
            
        except (OSError, IOError) as e:
            error_msg = f"Error scanning file {file_path}: {e}"
            logger.error(error_msg)
            self.scan_result.add_error(error_msg)
            return None
    
    def scan_directory(self, 
                      directory: Path, 
                      recursive: bool = True,
                      depth: int = 0) -> Generator[Tuple[Path, List[FileInfo]], None, None]:
        """
        Scan a directory and yield folder paths with their direct files.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            depth: Current recursion depth
            
        Yields:
            Tuples of (folder_path, list_of_files_in_folder)
        """
        if self.config.MAX_SCAN_DEPTH is not None and depth > self.config.MAX_SCAN_DEPTH:
            return
        
        try:
            files_in_folder = []
            subdirs = []
            
            for entry in os.scandir(directory):
                entry_path = Path(entry.path)
                
                if self.should_ignore(entry_path):
                    continue
                
                if entry.is_file(follow_symlinks=self.config.FOLLOW_SYMLINKS):
                    file_info = self.scan_file(entry_path)
                    if file_info:
                        files_in_folder.append(file_info)
                        self.scan_result.total_files += 1
                        self.scan_result.total_size += file_info.size
                
                elif entry.is_dir(follow_symlinks=self.config.FOLLOW_SYMLINKS) and recursive:
                    subdirs.append(entry_path)
            
            # Yield this folder's files
            yield (directory, files_in_folder)
            self.scan_result.total_folders += 1
            
            # Recursively scan subdirectories
            for subdir in subdirs:
                yield from self.scan_directory(subdir, recursive, depth + 1)
            
        except (OSError, IOError) as e:
            error_msg = f"Error scanning directory {directory}: {e}"
            logger.error(error_msg)
            self.scan_result.add_error(error_msg)
    
    def scan_tree(self, root_path: Path) -> FolderInfo:
        """
        Scan entire directory tree and build hierarchical structure.
        
        Args:
            root_path: Root directory to scan
            
        Returns:
            FolderInfo object representing the root with all subfolders
        """
        start_time = time.time()
        
        # Create root folder info
        root_folder = FolderInfo(
            path=root_path,
            name=root_path.name,
            parent_path=root_path.parent
        )
        
        # Build a map of path to FolderInfo
        folder_map = {root_path: root_folder}
        
        # Scan and build structure
        for folder_path, files in self.scan_directory(root_path):
            if folder_path not in folder_map:
                folder_info = FolderInfo(
                    path=folder_path,
                    name=folder_path.name,
                    parent_path=folder_path.parent
                )
                folder_map[folder_path] = folder_info
                
                # Link to parent if exists
                if folder_path.parent in folder_map:
                    folder_map[folder_path.parent].subfolders.append(folder_info)
            
            folder = folder_map[folder_path]
            folder.files = files
            folder.num_files = len(files)
            folder.total_size = sum(f.size for f in files)
        
        self.scan_result.scan_duration = time.time() - start_time
        
        return root_folder
    
    def scan_multiple(self, paths: List[Path]) -> List[FolderInfo]:
        """
        Scan multiple paths and return their information.
        
        Args:
            paths: List of paths to scan
            
        Returns:
            List of FolderInfo objects
        """
        results = []
        
        for path in paths:
            if path.is_file():
                # Single file - wrap in a pseudo-folder
                file_info = self.scan_file(path)
                if file_info:
                    folder = FolderInfo(
                        path=path.parent,
                        name=path.parent.name,
                        files=[file_info],
                        num_files=1,
                        total_size=file_info.size
                    )
                    results.append(folder)
            elif path.is_dir():
                folder = self.scan_tree(path)
                results.append(folder)
            else:
                error_msg = f"Path does not exist: {path}"
                logger.error(error_msg)
                self.scan_result.add_error(error_msg)
        
        return results
    
    def get_scan_result(self) -> ScanResult:
        """Get the current scan result with statistics."""
        return self.scan_result
    
    def reset(self):
        """Reset the scanner state for a new scan."""
        self.scan_result = ScanResult()


class ProgressScanner(FileScanner):
    """File scanner with progress reporting capabilities."""
    
    def __init__(self, config: Config = None, progress_callback=None):
        """
        Initialize scanner with progress callback.
        
        Args:
            config: Configuration object
            progress_callback: Function called with (current_count, current_path)
        """
        super().__init__(config)
        self.progress_callback = progress_callback
        self.file_count = 0
    
    def scan_file(self, file_path: Path) -> Optional[FileInfo]:
        """Scan file and report progress."""
        result = super().scan_file(file_path)
        
        if result and self.progress_callback:
            self.file_count += 1
            self.progress_callback(self.file_count, str(file_path))
        
        return result