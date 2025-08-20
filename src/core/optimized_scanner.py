"""Optimized file system scanning with async I/O, caching, and intelligent strategies."""

import os
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Generator, Optional, Tuple, List, Dict, Set, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import deque
import pickle

try:
    import aiofiles
    ASYNC_IO_AVAILABLE = True
except ImportError:
    ASYNC_IO_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, n=1):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

from ..config import Config
from ..storage.models import FileInfo, FolderInfo, ScanResult

logger = logging.getLogger(__name__)


@dataclass
class ScanCache:
    """Cache for scan results to avoid re-scanning unchanged directories."""
    folder_stats: Dict[str, Tuple[float, int, int]] = field(default_factory=dict)  # path -> (mtime, file_count, total_size)
    file_cache: Dict[str, FileInfo] = field(default_factory=dict)  # path -> FileInfo
    last_scan_time: Dict[str, float] = field(default_factory=dict)  # path -> timestamp
    
    def save(self, cache_path: Path):
        """Save cache to disk."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            logger.warning(f"Failed to save scan cache: {e}")
    
    @classmethod
    def load(cls, cache_path: Path) -> 'ScanCache':
        """Load cache from disk."""
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load scan cache: {e}")
        return cls()


class OptimizedFileScanner:
    """High-performance file scanner with multiple optimization strategies."""
    
    # Optimal batch sizes for different operations
    BATCH_SIZE = 1000
    PREFETCH_SIZE = 100
    
    def __init__(self, config: Config = None):
        """Initialize the optimized scanner."""
        self.config = config or Config()
        self.scan_result = ScanResult()
        
        # Initialize scan cache
        self.cache_path = Path.home() / '.pydeduper' / 'scan_cache.pkl'
        self.cache = ScanCache.load(self.cache_path)
        
        # Directory entry cache for repeated access
        self._dir_cache = {}
        self._dir_cache_size = 1000
        self._dir_cache_order = deque(maxlen=self._dir_cache_size)
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'dirs_scanned': 0,
            'files_scanned': 0,
            'scan_time': 0.0
        }
    
    def should_rescan(self, path: Path) -> bool:
        """Check if a directory needs rescanning based on modification time."""
        try:
            stat = path.stat()
            cache_key = str(path)
            
            if cache_key in self.cache.folder_stats:
                cached_mtime, _, _ = self.cache.folder_stats[cache_key]
                if stat.st_mtime <= cached_mtime:
                    self.stats['cache_hits'] += 1
                    return False
            
            self.stats['cache_misses'] += 1
            return True
            
        except Exception:
            return True
    
    def scan_directory_cached(self, directory: Path) -> Tuple[List[FileInfo], List[Path]]:
        """Scan directory with caching of results."""
        cache_key = str(directory)
        
        # Check if we can use cached results
        if not self.should_rescan(directory):
            # Return cached results
            cached_files = []
            subdirs = []
            
            for entry_path in self._get_cached_entries(directory):
                if entry_path.is_file():
                    if str(entry_path) in self.cache.file_cache:
                        cached_files.append(self.cache.file_cache[str(entry_path)])
                elif entry_path.is_dir():
                    subdirs.append(entry_path)
            
            if cached_files or subdirs:
                return cached_files, subdirs
        
        # Perform actual scan
        files = []
        subdirs = []
        
        try:
            # Use scandir for efficient directory traversal
            with os.scandir(directory) as entries:
                entry_list = list(entries)
                
                # Cache directory entries
                self._cache_dir_entries(directory, entry_list)
                
                for entry in entry_list:
                    entry_path = Path(entry.path)
                    
                    # Skip ignored files
                    if self._should_ignore(entry_path):
                        continue
                    
                    if entry.is_file(follow_symlinks=self.config.FOLLOW_SYMLINKS):
                        file_info = self._create_file_info_from_entry(entry)
                        if file_info:
                            files.append(file_info)
                            # Cache file info
                            self.cache.file_cache[str(entry_path)] = file_info
                            self.stats['files_scanned'] += 1
                    
                    elif entry.is_dir(follow_symlinks=self.config.FOLLOW_SYMLINKS):
                        subdirs.append(entry_path)
            
            # Update folder cache
            total_size = sum(f.size for f in files)
            self.cache.folder_stats[cache_key] = (
                directory.stat().st_mtime,
                len(files),
                total_size
            )
            self.stats['dirs_scanned'] += 1
            
        except (OSError, IOError) as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            self.scan_result.add_error(str(e))
        
        return files, subdirs
    
    def _get_cached_entries(self, directory: Path) -> List[Path]:
        """Get cached directory entries."""
        cache_key = str(directory)
        if cache_key in self._dir_cache:
            # Move to end of order (LRU)
            self._dir_cache_order.remove(cache_key)
            self._dir_cache_order.append(cache_key)
            return self._dir_cache[cache_key]
        return []
    
    def _cache_dir_entries(self, directory: Path, entries: List):
        """Cache directory entries with LRU eviction."""
        cache_key = str(directory)
        entry_paths = [Path(e.path) for e in entries]
        
        # LRU eviction if cache is full
        if len(self._dir_cache) >= self._dir_cache_size:
            oldest = self._dir_cache_order.popleft()
            del self._dir_cache[oldest]
        
        self._dir_cache[cache_key] = entry_paths
        self._dir_cache_order.append(cache_key)
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        name = path.name
        
        if self.config.IGNORE_HIDDEN and name.startswith('.'):
            return True
        
        for pattern in self.config.IGNORE_PATTERNS:
            if path.match(pattern):
                return True
        
        return False
    
    def _create_file_info_from_entry(self, entry: os.DirEntry) -> Optional[FileInfo]:
        """Create FileInfo from DirEntry with caching."""
        try:
            stat = entry.stat(follow_symlinks=self.config.FOLLOW_SYMLINKS)
            
            return FileInfo(
                path=Path(entry.path),
                name=entry.name,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime)
            )
        except Exception as e:
            logger.error(f"Error creating FileInfo for {entry.path}: {e}")
            return None
    
    def scan_parallel_batch(self, directories: List[Path], max_workers: int = None) -> Dict[Path, Tuple[List[FileInfo], List[Path]]]:
        """Scan multiple directories in parallel with batching."""
        max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        results = {}
        
        def scan_worker(directory: Path) -> Tuple[Path, List[FileInfo], List[Path]]:
            files, subdirs = self.scan_directory_cached(directory)
            return directory, files, subdirs
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process in batches for better memory management
            for i in range(0, len(directories), self.BATCH_SIZE):
                batch = directories[i:i + self.BATCH_SIZE]
                futures = [executor.submit(scan_worker, d) for d in batch]
                
                for future in futures:
                    try:
                        directory, files, subdirs = future.result()
                        results[directory] = (files, subdirs)
                    except Exception as e:
                        logger.error(f"Error in parallel scan: {e}")
        
        return results
    
    def scan_tree_optimized(self, root_path: Path, use_parallel: bool = True) -> FolderInfo:
        """Scan directory tree with optimizations."""
        start_time = time.time()
        
        # Create root folder
        root_folder = FolderInfo(
            path=root_path,
            name=root_path.name,
            parent_path=root_path.parent
        )
        
        # Use BFS for better cache locality
        folder_map = {root_path: root_folder}
        to_scan = deque([root_path])
        
        # Batch processing for parallel scanning
        batch_to_scan = []
        
        while to_scan or batch_to_scan:
            # Collect batch for parallel processing
            while to_scan and len(batch_to_scan) < self.BATCH_SIZE:
                batch_to_scan.append(to_scan.popleft())
            
            if batch_to_scan:
                if use_parallel and len(batch_to_scan) > 1:
                    # Parallel scan
                    scan_results = self.scan_parallel_batch(batch_to_scan)
                    
                    for directory, (files, subdirs) in scan_results.items():
                        folder = folder_map[directory]
                        folder.files = files
                        folder.num_files = len(files)
                        folder.total_size = sum(f.size for f in files)
                        
                        # Add subdirectories to scan queue
                        for subdir in subdirs:
                            if subdir not in folder_map:
                                subfolder = FolderInfo(
                                    path=subdir,
                                    name=subdir.name,
                                    parent_path=directory
                                )
                                folder_map[subdir] = subfolder
                                folder.subfolders.append(subfolder)
                                to_scan.append(subdir)
                else:
                    # Sequential scan for small batches
                    for directory in batch_to_scan:
                        files, subdirs = self.scan_directory_cached(directory)
                        
                        folder = folder_map[directory]
                        folder.files = files
                        folder.num_files = len(files)
                        folder.total_size = sum(f.size for f in files)
                        
                        for subdir in subdirs:
                            if subdir not in folder_map:
                                subfolder = FolderInfo(
                                    path=subdir,
                                    name=subdir.name,
                                    parent_path=directory
                                )
                                folder_map[subdir] = subfolder
                                folder.subfolders.append(subfolder)
                                to_scan.append(subdir)
                
                batch_to_scan.clear()
        
        self.scan_result.scan_duration = time.time() - start_time
        self.scan_result.total_folders = len(folder_map)
        self.scan_result.total_files = sum(f.num_files for f in folder_map.values())
        self.scan_result.total_size = sum(f.total_size for f in folder_map.values())
        
        # Save cache periodically
        self.save_cache()
        
        return root_folder
    
    def save_cache(self):
        """Save scan cache to disk."""
        self.cache.save(self.cache_path)
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache = ScanCache()
        self._dir_cache.clear()
        self._dir_cache_order.clear()
    
    def get_statistics(self) -> Dict:
        """Get scanning statistics."""
        return self.stats


class AsyncOptimizedScanner(OptimizedFileScanner):
    """Asynchronous version of the optimized scanner."""
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        self.semaphore = asyncio.Semaphore(100)  # Limit concurrent I/O operations
    
    async def scan_file_async(self, file_path: Path) -> Optional[FileInfo]:
        """Scan a single file asynchronously."""
        if not ASYNC_IO_AVAILABLE:
            return self._create_file_info_from_entry(file_path)
        
        async with self.semaphore:
            try:
                loop = asyncio.get_event_loop()
                stat = await loop.run_in_executor(None, file_path.stat)
                
                return FileInfo(
                    path=file_path,
                    name=file_path.name,
                    size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime)
                )
            except Exception as e:
                logger.error(f"Error scanning file async {file_path}: {e}")
                return None
    
    async def scan_directory_async(self, directory: Path) -> Tuple[List[FileInfo], List[Path]]:
        """Scan directory asynchronously."""
        if not self.should_rescan(directory):
            # Use cached results
            return self.scan_directory_cached(directory)
        
        files = []
        subdirs = []
        
        try:
            loop = asyncio.get_event_loop()
            
            # Run scandir in executor
            entries = await loop.run_in_executor(None, lambda: list(os.scandir(directory)))
            
            # Process entries concurrently
            file_tasks = []
            
            for entry in entries:
                entry_path = Path(entry.path)
                
                if self._should_ignore(entry_path):
                    continue
                
                if entry.is_file(follow_symlinks=self.config.FOLLOW_SYMLINKS):
                    task = self.scan_file_async(entry_path)
                    file_tasks.append(task)
                elif entry.is_dir(follow_symlinks=self.config.FOLLOW_SYMLINKS):
                    subdirs.append(entry_path)
            
            # Wait for all file scans to complete
            if file_tasks:
                file_results = await asyncio.gather(*file_tasks, return_exceptions=True)
                files = [f for f in file_results if f and not isinstance(f, Exception)]
            
            # Update cache
            cache_key = str(directory)
            total_size = sum(f.size for f in files)
            self.cache.folder_stats[cache_key] = (
                directory.stat().st_mtime,
                len(files),
                total_size
            )
            
            for file_info in files:
                self.cache.file_cache[str(file_info.path)] = file_info
            
        except Exception as e:
            logger.error(f"Error in async directory scan {directory}: {e}")
        
        return files, subdirs
    
    async def scan_tree_async(self, root_path: Path) -> FolderInfo:
        """Scan directory tree asynchronously."""
        start_time = time.time()
        
        root_folder = FolderInfo(
            path=root_path,
            name=root_path.name,
            parent_path=root_path.parent
        )
        
        folder_map = {root_path: root_folder}
        to_scan = deque([root_path])
        
        while to_scan:
            # Process directories in batches
            batch = []
            for _ in range(min(self.BATCH_SIZE, len(to_scan))):
                if to_scan:
                    batch.append(to_scan.popleft())
            
            if batch:
                # Scan batch concurrently
                scan_tasks = [self.scan_directory_async(d) for d in batch]
                results = await asyncio.gather(*scan_tasks, return_exceptions=True)
                
                for directory, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error scanning {directory}: {result}")
                        continue
                    
                    files, subdirs = result
                    folder = folder_map[directory]
                    folder.files = files
                    folder.num_files = len(files)
                    folder.total_size = sum(f.size for f in files)
                    
                    for subdir in subdirs:
                        if subdir not in folder_map:
                            subfolder = FolderInfo(
                                path=subdir,
                                name=subdir.name,
                                parent_path=directory
                            )
                            folder_map[subdir] = subfolder
                            folder.subfolders.append(subfolder)
                            to_scan.append(subdir)
        
        self.scan_result.scan_duration = time.time() - start_time
        self.scan_result.total_folders = len(folder_map)
        self.scan_result.total_files = sum(f.num_files for f in folder_map.values())
        self.scan_result.total_size = sum(f.total_size for f in folder_map.values())
        
        # Save cache
        await asyncio.get_event_loop().run_in_executor(None, self.save_cache)
        
        return root_folder


class IntelligentScanner(OptimizedFileScanner):
    """Scanner with intelligent strategies for different scenarios."""
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        self.size_threshold_for_sampling = 1000  # Sample if > 1000 files
    
    def scan_with_early_termination(self, root_path: Path) -> FolderInfo:
        """Scan with early termination for unique file sizes."""
        # Track file sizes globally for early duplicate detection
        size_map = {}
        unique_sizes = set()
        
        root_folder = self.scan_tree_optimized(root_path)
        
        # Build size map
        def collect_files(folder: FolderInfo):
            for file in folder.files:
                if file.size not in size_map:
                    size_map[file.size] = []
                size_map[file.size].append(file)
            
            for subfolder in folder.subfolders:
                collect_files(subfolder)
        
        collect_files(root_folder)
        
        # Mark files with unique sizes (no need for hashing)
        for size, files in size_map.items():
            if len(files) == 1:
                unique_sizes.add(size)
                files[0].is_unique = True  # Mark as unique
        
        logger.info(f"Found {len(unique_sizes)} files with unique sizes (no hashing needed)")
        
        return root_folder
    
    def scan_with_inode_detection(self, root_path: Path) -> FolderInfo:
        """Scan with hardlink detection using inodes."""
        inode_map = {}
        
        root_folder = self.scan_tree_optimized(root_path)
        
        # Detect hardlinks
        def process_folder(folder: FolderInfo):
            for file in folder.files:
                try:
                    stat = file.path.stat()
                    inode = (stat.st_dev, stat.st_ino)
                    
                    if inode in inode_map:
                        # This is a hardlink to an existing file
                        file.hardlink_to = inode_map[inode]
                    else:
                        inode_map[inode] = file.path
                        
                except Exception as e:
                    logger.debug(f"Failed to get inode for {file.path}: {e}")
            
            for subfolder in folder.subfolders:
                process_folder(subfolder)
        
        process_folder(root_folder)
        
        hardlinks = sum(1 for folder in self._iter_folders(root_folder) 
                       for file in folder.files if hasattr(file, 'hardlink_to'))
        
        if hardlinks > 0:
            logger.info(f"Detected {hardlinks} hardlinks")
        
        return root_folder
    
    def _iter_folders(self, folder: FolderInfo) -> Generator[FolderInfo, None, None]:
        """Iterate through all folders in tree."""
        yield folder
        for subfolder in folder.subfolders:
            yield from self._iter_folders(subfolder)