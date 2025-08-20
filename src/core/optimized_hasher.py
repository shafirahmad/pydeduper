"""Optimized file hashing with memory-mapped I/O, hardware acceleration, and adaptive strategies."""

import hashlib
import mmap
import os
import logging
import pickle
import time
from pathlib import Path
from typing import Optional, BinaryIO, Dict, Tuple, List
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

try:
    # Try to use hardware-accelerated hashing if available
    import hashlib
    from hashlib import blake2b, blake2s
    HARDWARE_ACCEL_AVAILABLE = True
except ImportError:
    HARDWARE_ACCEL_AVAILABLE = False

try:
    from pybloom_live import BloomFilter
    BLOOM_AVAILABLE = True
except ImportError:
    BLOOM_AVAILABLE = False
    
from ..config import Config, HashAlgorithm
from ..storage.models import FileInfo

logger = logging.getLogger(__name__)


@dataclass
class HashingStats:
    """Statistics for hash operations."""
    total_bytes_processed: int = 0
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    mmap_used: int = 0
    streaming_used: int = 0
    
    @property
    def throughput_mbps(self) -> float:
        """Calculate throughput in MB/s."""
        if self.total_time == 0:
            return 0
        return (self.total_bytes_processed / (1024 * 1024)) / self.total_time


class OptimizedFileHasher:
    """High-performance file hasher with multiple optimization strategies."""
    
    # Optimal chunk sizes for different file size ranges
    CHUNK_SIZE_MAP = {
        1024 * 1024: 4096,           # < 1MB: 4KB chunks
        10 * 1024 * 1024: 8192,      # < 10MB: 8KB chunks  
        100 * 1024 * 1024: 32768,    # < 100MB: 32KB chunks
        1024 * 1024 * 1024: 65536,   # < 1GB: 64KB chunks
        float('inf'): 131072         # >= 1GB: 128KB chunks
    }
    
    # Memory mapping threshold
    MMAP_THRESHOLD = 10 * 1024 * 1024  # Use mmap for files > 10MB
    
    def __init__(self, config: Config = None):
        """Initialize the optimized hasher."""
        self.config = config or Config()
        self.algorithm = self.config.DEFAULT_HASH_ALGORITHM
        self.stats = HashingStats()
        
        # Initialize persistent cache
        self.cache_file = Path.home() / '.pydeduper' / 'hash_cache.pkl'
        self.persistent_cache = self._load_persistent_cache()
        
        # Initialize bloom filter for quick checks
        if BLOOM_AVAILABLE:
            self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)
        else:
            self.bloom_filter = None
            
        # LRU cache for recent hashes
        self._memory_cache = {}
        self._cache_order = []
        self.max_cache_size = 10000
    
    def _load_persistent_cache(self) -> Dict[str, Tuple[str, float, int]]:
        """Load persistent cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_persistent_cache(self):
        """Save persistent cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.persistent_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_optimal_chunk_size(self, file_size: int) -> int:
        """Determine optimal chunk size based on file size."""
        for threshold, chunk_size in self.CHUNK_SIZE_MAP.items():
            if file_size < threshold:
                return chunk_size
        return 131072  # Default to 128KB
    
    def _get_hash_object(self, use_hardware: bool = True):
        """Get hash object with hardware acceleration if available."""
        if use_hardware and HARDWARE_ACCEL_AVAILABLE:
            # Blake2 is often hardware accelerated and faster than SHA
            if self.algorithm == HashAlgorithm.SHA256:
                return blake2b(digest_size=32)
            elif self.algorithm == HashAlgorithm.SHA512:
                return blake2b(digest_size=64)
        
        # Fallback to standard algorithms
        algorithm_map = {
            HashAlgorithm.MD5: hashlib.md5,
            HashAlgorithm.SHA1: hashlib.sha1,
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA512: hashlib.sha512
        }
        return algorithm_map[self.algorithm]()
    
    def _check_cache(self, file_path: Path) -> Optional[str]:
        """Check if file hash is in cache."""
        try:
            stat = file_path.stat()
            cache_key = str(file_path)
            
            # Check memory cache first
            if cache_key in self._memory_cache:
                cached_hash, cached_mtime, cached_size = self._memory_cache[cache_key]
                if cached_mtime == stat.st_mtime and cached_size == stat.st_size:
                    self.stats.cache_hits += 1
                    return cached_hash
            
            # Check persistent cache
            if cache_key in self.persistent_cache:
                cached_hash, cached_mtime, cached_size = self.persistent_cache[cache_key]
                if cached_mtime == stat.st_mtime and cached_size == stat.st_size:
                    self.stats.cache_hits += 1
                    # Promote to memory cache
                    self._update_memory_cache(cache_key, cached_hash, cached_mtime, cached_size)
                    return cached_hash
            
            self.stats.cache_misses += 1
            return None
            
        except Exception as e:
            logger.debug(f"Cache check failed for {file_path}: {e}")
            return None
    
    def _update_memory_cache(self, key: str, hash_value: str, mtime: float, size: int):
        """Update memory cache with LRU eviction."""
        if key in self._memory_cache:
            self._cache_order.remove(key)
        elif len(self._memory_cache) >= self.max_cache_size:
            # Evict oldest entry
            oldest = self._cache_order.pop(0)
            del self._memory_cache[oldest]
        
        self._memory_cache[key] = (hash_value, mtime, size)
        self._cache_order.append(key)
    
    def _store_cache(self, file_path: Path, hash_value: str):
        """Store hash in both memory and persistent cache."""
        try:
            stat = file_path.stat()
            cache_key = str(file_path)
            cache_value = (hash_value, stat.st_mtime, stat.st_size)
            
            # Update memory cache
            self._update_memory_cache(cache_key, hash_value, stat.st_mtime, stat.st_size)
            
            # Update persistent cache
            self.persistent_cache[cache_key] = cache_value
            
            # Update bloom filter if available
            if self.bloom_filter:
                self.bloom_filter.add(hash_value)
                
        except Exception as e:
            logger.debug(f"Failed to cache hash for {file_path}: {e}")
    
    def hash_file_mmap(self, file_path: Path) -> Optional[str]:
        """Hash file using memory-mapped I/O for large files."""
        start_time = time.time()
        
        try:
            file_size = file_path.stat().st_size
            
            # Check cache first
            cached_hash = self._check_cache(file_path)
            if cached_hash:
                return cached_hash
            
            hash_obj = self._get_hash_object()
            
            with open(file_path, 'rb') as f:
                if file_size > self.MMAP_THRESHOLD and file_size > 0:
                    # Use memory mapping for large files
                    self.stats.mmap_used += 1
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                        # Process in optimal chunks
                        chunk_size = self._get_optimal_chunk_size(file_size)
                        for i in range(0, file_size, chunk_size):
                            chunk = mmapped_file[i:i + chunk_size]
                            hash_obj.update(chunk)
                else:
                    # Use regular I/O for small files
                    self.stats.streaming_used += 1
                    chunk_size = self._get_optimal_chunk_size(file_size)
                    while chunk := f.read(chunk_size):
                        hash_obj.update(chunk)
            
            hash_value = hash_obj.hexdigest()
            
            # Update statistics
            self.stats.total_bytes_processed += file_size
            self.stats.total_time += time.time() - start_time
            
            # Store in cache
            self._store_cache(file_path, hash_value)
            
            return hash_value
            
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None
    
    def hash_file_adaptive(self, file_path: Path) -> Optional[str]:
        """Adaptively choose hashing strategy based on file characteristics."""
        try:
            file_size = file_path.stat().st_size
            
            # Very small files: read entirely into memory
            if file_size < 1024 * 1024:  # < 1MB
                return self.hash_file_memory(file_path)
            
            # Large files: use memory mapping
            elif file_size > self.MMAP_THRESHOLD:
                return self.hash_file_mmap(file_path)
            
            # Medium files: use optimized streaming
            else:
                return self.hash_file_streaming(file_path)
                
        except Exception as e:
            logger.error(f"Error in adaptive hashing for {file_path}: {e}")
            return None
    
    def hash_file_memory(self, file_path: Path) -> Optional[str]:
        """Hash small files entirely in memory."""
        start_time = time.time()
        
        try:
            # Check cache
            cached_hash = self._check_cache(file_path)
            if cached_hash:
                return cached_hash
            
            # Read entire file into memory
            with open(file_path, 'rb') as f:
                content = f.read()
            
            hash_obj = self._get_hash_object()
            hash_obj.update(content)
            hash_value = hash_obj.hexdigest()
            
            # Update statistics
            self.stats.total_bytes_processed += len(content)
            self.stats.total_time += time.time() - start_time
            
            # Store in cache
            self._store_cache(file_path, hash_value)
            
            return hash_value
            
        except Exception as e:
            logger.error(f"Error hashing file in memory {file_path}: {e}")
            return None
    
    def hash_file_streaming(self, file_path: Path) -> Optional[str]:
        """Hash file using optimized streaming."""
        start_time = time.time()
        
        try:
            file_size = file_path.stat().st_size
            
            # Check cache
            cached_hash = self._check_cache(file_path)
            if cached_hash:
                return cached_hash
            
            hash_obj = self._get_hash_object()
            chunk_size = self._get_optimal_chunk_size(file_size)
            
            with open(file_path, 'rb', buffering=chunk_size) as f:
                while chunk := f.read(chunk_size):
                    hash_obj.update(chunk)
            
            hash_value = hash_obj.hexdigest()
            
            # Update statistics
            self.stats.total_bytes_processed += file_size
            self.stats.total_time += time.time() - start_time
            self.stats.streaming_used += 1
            
            # Store in cache
            self._store_cache(file_path, hash_value)
            
            return hash_value
            
        except Exception as e:
            logger.error(f"Error in streaming hash for {file_path}: {e}")
            return None
    
    def hash_file_sampling(self, file_path: Path, sample_size: int = 1024 * 1024) -> Optional[str]:
        """Hash file using intelligent sampling for very large files."""
        try:
            file_size = file_path.stat().st_size
            
            if file_size <= sample_size * 3:
                # File is small enough to hash completely
                return self.hash_file_adaptive(file_path)
            
            hash_obj = self._get_hash_object()
            
            with open(file_path, 'rb') as f:
                # Sample from beginning
                hash_obj.update(f.read(sample_size))
                
                # Sample from multiple positions using golden ratio
                golden_ratio = 1.618033988749895
                num_samples = min(10, file_size // sample_size)
                
                for i in range(1, num_samples):
                    position = int((file_size * i * golden_ratio) % file_size)
                    f.seek(position)
                    hash_obj.update(f.read(min(sample_size, file_size - position)))
                
                # Sample from end
                f.seek(max(0, file_size - sample_size))
                hash_obj.update(f.read(sample_size))
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error in sampling hash for {file_path}: {e}")
            return None
    
    def check_bloom_filter(self, hash_value: str) -> bool:
        """Check if hash might be a duplicate using bloom filter."""
        if self.bloom_filter:
            return hash_value in self.bloom_filter
        return True  # Conservative: assume it might be duplicate if no bloom filter
    
    def batch_hash_files(self, files: List[Path], use_sampling: bool = False) -> Dict[str, str]:
        """Hash multiple files in batch with optimal strategy."""
        results = {}
        
        # Sort files by size for better cache locality
        sorted_files = sorted(files, key=lambda f: f.stat().st_size if f.exists() else 0)
        
        for file_path in sorted_files:
            if use_sampling and file_path.stat().st_size > 100 * 1024 * 1024:
                hash_value = self.hash_file_sampling(file_path)
            else:
                hash_value = self.hash_file_adaptive(file_path)
            
            if hash_value:
                results[str(file_path)] = hash_value
        
        return results
    
    def get_statistics(self) -> HashingStats:
        """Get hashing statistics."""
        return self.stats
    
    def save_cache(self):
        """Save persistent cache to disk."""
        self._save_persistent_cache()
    
    def clear_cache(self):
        """Clear all caches."""
        self._memory_cache.clear()
        self._cache_order.clear()
        self.persistent_cache.clear()
        if self.bloom_filter:
            self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001) if BLOOM_AVAILABLE else None


class ParallelOptimizedHasher(OptimizedFileHasher):
    """Parallel version of optimized hasher with work stealing."""
    
    def __init__(self, config: Config = None, max_workers: int = None):
        super().__init__(config)
        self.max_workers = max_workers or os.cpu_count()
    
    def parallel_hash_files(self, files: List[Path]) -> Dict[str, str]:
        """Hash files in parallel with work stealing queue."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from queue import Queue
        
        results = {}
        work_queue = Queue()
        
        # Group files by size for better load balancing
        small_files = []
        medium_files = []
        large_files = []
        
        for file in files:
            try:
                size = file.stat().st_size
                if size < 1024 * 1024:  # < 1MB
                    small_files.append(file)
                elif size < 100 * 1024 * 1024:  # < 100MB
                    medium_files.append(file)
                else:
                    large_files.append(file)
            except:
                continue
        
        # Process in order: small files first for quick wins
        all_files = small_files + medium_files + large_files
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit initial batch
            futures = {}
            for file in all_files[:self.max_workers * 2]:
                future = executor.submit(self.hash_file_adaptive, file)
                futures[future] = file
            
            remaining = all_files[self.max_workers * 2:]
            
            # Process with work stealing
            for future in as_completed(futures):
                file = futures[future]
                try:
                    hash_value = future.result()
                    if hash_value:
                        results[str(file)] = hash_value
                    
                    # Submit next file if available
                    if remaining:
                        next_file = remaining.pop(0)
                        new_future = executor.submit(self.hash_file_adaptive, next_file)
                        futures[new_future] = next_file
                        
                except Exception as e:
                    logger.error(f"Error hashing {file}: {e}")
        
        return results