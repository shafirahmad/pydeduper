"""Parallel hash calculation with progress reporting and optimization."""

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Queue, Empty
import multiprocessing as mp

try:
    from tqdm import tqdm
except ImportError:
    # Fallback implementation
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

from ..config import Config, HashAlgorithm
from ..storage.models import FileInfo
from .hasher import FileHasher, HashCache

logger = logging.getLogger(__name__)


@dataclass
class HashTask:
    """Represents a hash calculation task."""
    file_info: FileInfo
    task_type: str  # 'quick' or 'full' or 'partial'
    priority: int = 0  # Higher priority processed first
    
    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


@dataclass
class HashResult:
    """Result of a hash calculation task."""
    file_info: FileInfo
    hash_value: Optional[str]
    task_type: str
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0


def hash_worker(task: HashTask, config: Config) -> HashResult:
    """
    Worker function for hash calculation.
    
    Args:
        task: Hash task to process
        config: Configuration object
        
    Returns:
        HashResult with calculated hash or error
    """
    start_time = time.time()
    hasher = FileHasher(config)
    
    try:
        if task.task_type == 'quick':
            hash_value = hasher.quick_hash(task.file_info.path)
        elif task.task_type == 'partial':
            hash_value = hasher.hash_file_partial(task.file_info.path)
        else:  # 'full'
            hash_value = hasher.hash_file(task.file_info.path)
        
        processing_time = time.time() - start_time
        
        return HashResult(
            file_info=task.file_info,
            hash_value=hash_value,
            task_type=task.task_type,
            success=hash_value is not None,
            processing_time=processing_time
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        return HashResult(
            file_info=task.file_info,
            hash_value=None,
            task_type=task.task_type,
            success=False,
            error_message=str(e),
            processing_time=processing_time
        )


class ParallelHasher:
    """Parallel hash calculator with progress reporting and caching."""
    
    def __init__(self, config: Config = None, max_workers: int = None, 
                 use_process_pool: bool = False):
        """
        Initialize parallel hasher.
        
        Args:
            config: Configuration object
            max_workers: Maximum number of workers
            use_process_pool: Whether to use process pool instead of thread pool
        """
        self.config = config or Config()
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.use_process_pool = use_process_pool
        self.cache = HashCache()
        self._cancelled = threading.Event()
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    def cancel(self):
        """Cancel all pending hash operations."""
        self._cancelled.set()
    
    def is_cancelled(self) -> bool:
        """Check if operations were cancelled."""
        return self._cancelled.is_set()
    
    def _check_cache(self, file_info: FileInfo, task_type: str) -> Optional[str]:
        """Check cache for existing hash."""
        mtime = file_info.last_modified.timestamp() if file_info.last_modified else None
        
        if task_type == 'quick':
            hash_value = self.cache.get_quick_hash(file_info.path, mtime)
        else:
            hash_value = self.cache.get_hash(file_info.path, mtime)
        
        if hash_value:
            self.stats['cache_hits'] += 1
            return hash_value
        
        return None
    
    def _store_cache(self, result: HashResult):
        """Store result in cache."""
        if not result.success or not result.hash_value:
            return
        
        mtime = (result.file_info.last_modified.timestamp() 
                if result.file_info.last_modified else None)
        
        if result.task_type == 'quick':
            self.cache.set_quick_hash(result.file_info.path, result.hash_value, mtime)
        else:
            self.cache.set_hash(result.file_info.path, result.hash_value, mtime)
    
    def calculate_hashes_parallel(self, 
                                 files: List[FileInfo], 
                                 task_type: str = 'full',
                                 progress_callback: Optional[Callable] = None) -> Dict[str, str]:
        """
        Calculate hashes for multiple files in parallel.
        
        Args:
            files: List of FileInfo objects
            task_type: Type of hash to calculate ('quick', 'full', 'partial')
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary mapping file path to hash value
        """
        if not files:
            return {}
        
        results = {}
        start_time = time.time()
        
        # Create tasks with priority (smaller files first for quick completion)
        tasks = []
        for file_info in files:
            # Check cache first
            cached_hash = self._check_cache(file_info, task_type)
            if cached_hash:
                results[str(file_info.path)] = cached_hash
                continue
            
            # Create task with priority based on file size (smaller = higher priority)
            priority = max(0, 1000000 - file_info.size)  # Reverse priority
            task = HashTask(file_info, task_type, priority)
            tasks.append(task)
        
        if not tasks:
            # All results were cached
            return results
        
        # Sort tasks by priority
        tasks.sort()
        
        # Progress tracking
        completed = len(results)  # Count cached results
        total = len(files)
        
        progress_bar = None
        if progress_callback is None:
            progress_bar = tqdm(
                total=total,
                desc=f"Calculating {task_type} hashes",
                unit='files',
                initial=completed
            )
        
        try:
            # Choose executor type
            executor_class = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit tasks
                future_to_task = {
                    executor.submit(hash_worker, task, self.config): task 
                    for task in tasks
                }
                
                # Process results as they complete
                for future in as_completed(future_to_task):
                    if self.is_cancelled():
                        break
                    
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        
                        # Update statistics
                        self.stats['files_processed'] += 1
                        self.stats['total_time'] += result.processing_time
                        
                        if result.success and result.hash_value:
                            results[str(result.file_info.path)] = result.hash_value
                            
                            # Update file info
                            if task_type == 'quick':
                                result.file_info.quick_hash = result.hash_value
                            else:
                                result.file_info.hash = result.hash_value
                            
                            # Store in cache
                            self._store_cache(result)
                        
                        else:
                            self.stats['errors'] += 1
                            if result.error_message:
                                logger.error(f"Hash calculation failed for {result.file_info.path}: {result.error_message}")
                        
                        # Update progress
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total, str(task.file_info.path))
                        elif progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_description(
                                f"Calculating {task_type} hashes [{completed}/{total}]"
                            )
                    
                    except Exception as e:
                        self.stats['errors'] += 1
                        logger.error(f"Error processing hash task: {e}")
                        completed += 1
                        if progress_bar:
                            progress_bar.update(1)
        
        finally:
            if progress_bar:
                progress_bar.close()
        
        return results
    
    def calculate_hashes_by_size_groups(self, 
                                       size_groups: Dict[int, List[FileInfo]],
                                       progress_callback: Optional[Callable] = None) -> Dict[str, List[FileInfo]]:
        """
        Calculate hashes for files grouped by size, with optimization.
        
        Args:
            size_groups: Dictionary mapping size to list of files
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary mapping hash to list of files with that hash
        """
        hash_groups = {}
        total_files = sum(len(files) for files in size_groups.values())
        processed_files = 0
        
        # Create main progress bar
        main_pbar = tqdm(
            total=len(size_groups),
            desc="Processing size groups",
            unit='groups'
        )
        
        try:
            for size, files in size_groups.items():
                if self.is_cancelled():
                    break
                
                group_desc = f"Size {size} bytes ({len(files)} files)"
                main_pbar.set_description(group_desc)
                
                if size > self.config.QUICK_HASH_SIZE * 3:
                    # Large files: use two-stage hashing
                    hash_results = self._two_stage_hashing(files, progress_callback)
                else:
                    # Small files: direct full hashing
                    hash_results = self.calculate_hashes_parallel(files, 'full', progress_callback)
                
                # Group by hash
                for file_path, hash_value in hash_results.items():
                    if hash_value not in hash_groups:
                        hash_groups[hash_value] = []
                    
                    # Find the file info
                    file_info = next((f for f in files if str(f.path) == file_path), None)
                    if file_info:
                        hash_groups[hash_value].append(file_info)
                
                processed_files += len(files)
                main_pbar.update(1)
        
        finally:
            main_pbar.close()
        
        # Filter out hashes with only one file
        return {h: files for h, files in hash_groups.items() if len(files) > 1}
    
    def _two_stage_hashing(self, files: List[FileInfo], 
                          progress_callback: Optional[Callable] = None) -> Dict[str, str]:
        """
        Two-stage hashing: quick hash first, then full hash for matches.
        
        Args:
            files: List of files to hash
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary mapping file path to full hash
        """
        # Stage 1: Quick hashes
        quick_hashes = self.calculate_hashes_parallel(files, 'quick')
        
        # Group by quick hash
        quick_groups = {}
        for file in files:
            quick_hash = quick_hashes.get(str(file.path))
            if quick_hash:
                if quick_hash not in quick_groups:
                    quick_groups[quick_hash] = []
                quick_groups[quick_hash].append(file)
        
        # Stage 2: Full hashes for files with duplicate quick hashes
        full_hashes = {}
        for quick_hash, group_files in quick_groups.items():
            if len(group_files) > 1:
                # Multiple files with same quick hash - calculate full hashes
                group_full_hashes = self.calculate_hashes_parallel(group_files, 'full')
                full_hashes.update(group_full_hashes)
            else:
                # Single file with unique quick hash - can't be duplicate
                # But we still need full hash for completeness
                group_full_hashes = self.calculate_hashes_parallel(group_files, 'full')
                full_hashes.update(group_full_hashes)
        
        return full_hashes
    
    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['files_processed'] > 0:
            stats['avg_time_per_file'] = stats['total_time'] / stats['files_processed']
            stats['cache_hit_rate'] = (stats['cache_hits'] / 
                                     (stats['cache_hits'] + stats['files_processed']))
        return stats
    
    def clear_cache(self):
        """Clear the hash cache."""
        self.cache.clear()


class AdaptiveHasher(ParallelHasher):
    """Adaptive hasher that adjusts strategy based on file characteristics."""
    
    def __init__(self, config: Config = None, max_workers: int = None):
        super().__init__(config, max_workers)
        self.performance_history = {}  # Track performance for different strategies
    
    def choose_hash_strategy(self, files: List[FileInfo]) -> str:
        """
        Choose optimal hashing strategy based on file characteristics.
        
        Args:
            files: List of files to analyze
            
        Returns:
            Optimal strategy ('quick', 'full', 'partial')
        """
        if not files:
            return 'full'
        
        # Analyze file characteristics
        total_size = sum(f.size for f in files)
        avg_size = total_size / len(files)
        max_size = max(f.size for f in files)
        
        # Decision logic
        if avg_size < self.config.QUICK_HASH_SIZE:
            return 'full'  # Small files, full hash is fast
        elif max_size > 100 * 1024 * 1024:  # > 100MB
            return 'partial'  # Very large files, use sampling
        else:
            return 'quick'  # Medium files, quick hash first
    
    def calculate_hashes_adaptive(self, files: List[FileInfo],
                                 progress_callback: Optional[Callable] = None) -> Dict[str, str]:
        """
        Calculate hashes using adaptive strategy selection.
        
        Args:
            files: List of files to hash
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary mapping file path to hash value
        """
        strategy = self.choose_hash_strategy(files)
        logger.info(f"Using {strategy} hash strategy for {len(files)} files")
        
        return self.calculate_hashes_parallel(files, strategy, progress_callback)