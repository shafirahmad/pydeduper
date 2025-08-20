"""Parallel file system scanning with progress reporting."""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Generator, Optional, Tuple, List, Callable, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import multiprocessing as mp

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
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
            print()  # New line at end
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

from ..config import Config
from ..storage.models import FileInfo, FolderInfo, ScanResult
from .scanner import FileScanner

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Handles progress reporting with multiple progress bars."""
    
    def __init__(self, enable_progress: bool = True):
        self.enable_progress = enable_progress
        self.progress_bars = {}
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
    
    def create_progress_bar(self, name: str, total: int, desc: str = None) -> Optional[tqdm]:
        """Create a new progress bar."""
        if not self.enable_progress:
            return None
        
        with self.lock:
            pbar = tqdm(
                total=total,
                desc=desc or name,
                unit='files',
                unit_scale=True,
                leave=True,
                position=len(self.progress_bars)
            )
            self.progress_bars[name] = pbar
            return pbar
    
    def update_progress(self, name: str, increment: int = 1, description: str = None):
        """Update progress for a specific progress bar."""
        if not self.enable_progress or name not in self.progress_bars:
            return
        
        with self.lock:
            pbar = self.progress_bars[name]
            pbar.update(increment)
            if description:
                pbar.set_description(description)
    
    def close_progress_bar(self, name: str):
        """Close and remove a progress bar."""
        if not self.enable_progress or name not in self.progress_bars:
            return
        
        with self.lock:
            pbar = self.progress_bars[name]
            pbar.close()
            del self.progress_bars[name]
    
    def cancel(self):
        """Signal cancellation to all operations."""
        self.cancelled.set()
        for pbar in self.progress_bars.values():
            pbar.close()
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self.cancelled.is_set()


def scan_directory_worker(args: Tuple[Path, Config, bool]) -> Tuple[Path, List[FileInfo], List[str]]:
    """Worker function for parallel directory scanning."""
    directory, config, follow_symlinks = args
    files = []
    errors = []
    
    try:
        for entry in os.scandir(directory):
            entry_path = Path(entry.path)
            
            # Check ignore patterns
            if any(entry_path.match(pattern) for pattern in config.IGNORE_PATTERNS):
                continue
            
            if config.IGNORE_HIDDEN and entry_path.name.startswith('.'):
                continue
            
            if entry.is_file(follow_symlinks=follow_symlinks):
                try:
                    stat = entry_path.stat()
                    file_info = FileInfo(
                        path=entry_path,
                        name=entry_path.name,
                        size=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime)
                    )
                    files.append(file_info)
                except (OSError, IOError) as e:
                    errors.append(f"Error scanning file {entry_path}: {e}")
    
    except (OSError, IOError) as e:
        errors.append(f"Error scanning directory {directory}: {e}")
    
    return directory, files, errors


class ParallelScanner(FileScanner):
    """Enhanced file scanner with parallel processing and progress reporting."""
    
    def __init__(self, config: Config = None, max_workers: int = None):
        """
        Initialize parallel scanner.
        
        Args:
            config: Configuration object
            max_workers: Maximum number of worker threads (default: CPU count)
        """
        super().__init__(config)
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.progress_reporter = ProgressReporter()
        self._cancel_event = threading.Event()
    
    def set_progress_enabled(self, enabled: bool):
        """Enable or disable progress reporting."""
        self.progress_reporter.enable_progress = enabled
    
    def cancel(self):
        """Cancel the current scanning operation."""
        self._cancel_event.set()
        self.progress_reporter.cancel()
    
    def is_cancelled(self) -> bool:
        """Check if scanning was cancelled."""
        return self._cancel_event.is_set()
    
    def count_total_items(self, paths: List[Path]) -> int:
        """Count total directories to scan for progress reporting."""
        total = 0
        for path in paths:
            if path.is_dir():
                try:
                    for root, dirs, files in os.walk(path):
                        if self.is_cancelled():
                            break
                        total += 1
                        # Apply ignore patterns to directories
                        dirs[:] = [d for d in dirs if not any(Path(root, d).match(pattern) 
                                                            for pattern in self.config.IGNORE_PATTERNS)]
                except (OSError, IOError):
                    pass
        return total
    
    def scan_directories_parallel(self, directories: List[Path]) -> Generator[Tuple[Path, List[FileInfo]], None, None]:
        """
        Scan multiple directories in parallel.
        
        Args:
            directories: List of directories to scan
            
        Yields:
            Tuples of (directory_path, list_of_files)
        """
        if not directories:
            return
        
        # Create progress bar
        pbar = self.progress_reporter.create_progress_bar(
            "scanning", 
            len(directories), 
            "Scanning directories"
        )
        
        # Prepare worker arguments
        worker_args = [
            (directory, self.config, self.config.FOLLOW_SYMLINKS)
            for directory in directories
        ]
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_dir = {
                    executor.submit(scan_directory_worker, args): args[0] 
                    for args in worker_args
                }
                
                # Process completed tasks
                for future in as_completed(future_to_dir):
                    if self.is_cancelled():
                        break
                    
                    directory = future_to_dir[future]
                    
                    try:
                        dir_path, files, errors = future.result()
                        
                        # Update scan result
                        self.scan_result.total_files += len(files)
                        self.scan_result.total_folders += 1
                        self.scan_result.total_size += sum(f.size for f in files)
                        self.scan_result.errors.extend(errors)
                        
                        # Update progress
                        self.progress_reporter.update_progress(
                            "scanning", 
                            1, 
                            f"Scanned {self.scan_result.total_files} files"
                        )
                        
                        yield dir_path, files
                        
                    except Exception as e:
                        error_msg = f"Error processing directory {directory}: {e}"
                        logger.error(error_msg)
                        self.scan_result.add_error(error_msg)
                        
                        self.progress_reporter.update_progress("scanning", 1)
        
        finally:
            self.progress_reporter.close_progress_bar("scanning")
    
    def scan_tree_parallel(self, root_path: Path) -> FolderInfo:
        """
        Scan directory tree with parallel processing and progress reporting.
        
        Args:
            root_path: Root directory to scan
            
        Returns:
            FolderInfo object with complete hierarchy
        """
        start_time = time.time()
        
        # First pass: collect all directories
        all_directories = []
        directory_to_parent = {}
        
        print("Analyzing directory structure...")
        try:
            for root, dirs, files in os.walk(root_path):
                if self.is_cancelled():
                    break
                
                root_path_obj = Path(root)
                all_directories.append(root_path_obj)
                
                # Apply ignore patterns to subdirectories
                dirs[:] = [d for d in dirs if not any(Path(root, d).match(pattern) 
                                                    for pattern in self.config.IGNORE_PATTERNS)]
                
                # Track parent-child relationships
                for dirname in dirs:
                    child_path = root_path_obj / dirname
                    directory_to_parent[child_path] = root_path_obj
        
        except (OSError, IOError) as e:
            error_msg = f"Error analyzing directory structure: {e}"
            logger.error(error_msg)
            self.scan_result.add_error(error_msg)
        
        if self.is_cancelled():
            return FolderInfo(path=root_path, name=root_path.name)
        
        print(f"Found {len(all_directories)} directories to scan")
        
        # Second pass: scan all directories in parallel
        folder_map = {}
        
        for dir_path, files in self.scan_directories_parallel(all_directories):
            if self.is_cancelled():
                break
            
            # Create FolderInfo
            folder_info = FolderInfo(
                path=dir_path,
                name=dir_path.name,
                parent_path=directory_to_parent.get(dir_path),
                files=files,
                num_files=len(files),
                total_size=sum(f.size for f in files)
            )
            
            folder_map[dir_path] = folder_info
        
        # Third pass: build hierarchy
        root_folder = folder_map.get(root_path)
        if not root_folder:
            root_folder = FolderInfo(path=root_path, name=root_path.name)
        
        for folder_path, folder in folder_map.items():
            parent_path = directory_to_parent.get(folder_path)
            if parent_path and parent_path in folder_map:
                parent_folder = folder_map[parent_path]
                parent_folder.subfolders.append(folder)
        
        self.scan_result.scan_duration = time.time() - start_time
        
        return root_folder
    
    def scan_multiple_parallel(self, paths: List[Path], callback: Callable[[str, int], None] = None) -> List[FolderInfo]:
        """
        Scan multiple paths in parallel with progress reporting.
        
        Args:
            paths: List of paths to scan
            callback: Optional callback for progress updates
            
        Returns:
            List of FolderInfo objects
        """
        results = []
        
        # Create main progress bar
        main_pbar = self.progress_reporter.create_progress_bar(
            "main", 
            len(paths), 
            "Processing paths"
        )
        
        try:
            for i, path in enumerate(paths):
                if self.is_cancelled():
                    break
                
                if callback:
                    callback(f"Processing {path}", i)
                
                if path.is_file():
                    # Single file
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
                    # Directory tree
                    folder = self.scan_tree_parallel(path)
                    results.append(folder)
                
                else:
                    error_msg = f"Path does not exist: {path}"
                    logger.error(error_msg)
                    self.scan_result.add_error(error_msg)
                
                self.progress_reporter.update_progress(
                    "main", 
                    1, 
                    f"Completed {i+1}/{len(paths)} paths"
                )
        
        finally:
            self.progress_reporter.close_progress_bar("main")
        
        return results


class BatchProgressReporter:
    """Progress reporter for batch operations with ETA calculation."""
    
    def __init__(self, total_items: int, operation_name: str = "Processing"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.processed = 0
        self.start_time = time.time()
        self.last_update = 0
        self.pbar = None
        
        if total_items > 0:
            self.pbar = tqdm(
                total=total_items,
                desc=operation_name,
                unit='items',
                unit_scale=True
            )
    
    def update(self, increment: int = 1, status: str = None):
        """Update progress and optionally set status."""
        if not self.pbar:
            return
        
        self.processed += increment
        self.pbar.update(increment)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.processed > 0:
            rate = self.processed / elapsed
            remaining = self.total_items - self.processed
            eta = remaining / rate if rate > 0 else 0
            
            if status:
                desc = f"{self.operation_name} - {status} (ETA: {eta:.0f}s)"
            else:
                desc = f"{self.operation_name} (ETA: {eta:.0f}s)"
            
            self.pbar.set_description(desc)
    
    def close(self):
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage and integration functions
def create_parallel_scanner(config: Config = None, max_workers: int = None, 
                          enable_progress: bool = True) -> ParallelScanner:
    """
    Factory function to create a configured parallel scanner.
    
    Args:
        config: Configuration object
        max_workers: Maximum worker threads
        enable_progress: Whether to show progress bars
        
    Returns:
        Configured ParallelScanner instance
    """
    scanner = ParallelScanner(config, max_workers)
    scanner.set_progress_enabled(enable_progress)
    return scanner