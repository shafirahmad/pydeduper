"""Integrated optimized analyzer combining all performance improvements."""

import logging
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import multiprocessing as mp

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

from ..config import Config
from ..storage.models import FileInfo, FolderInfo, DuplicateGroup, AnalysisResult
from .optimized_hasher import ParallelOptimizedHasher
from .optimized_scanner import IntelligentScanner, AsyncOptimizedScanner
from ..storage.optimized_database import OptimizedDatabase, MemoryIndex

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStrategy:
    """Configuration for optimization strategies."""
    use_async_io: bool = True
    use_memory_mapping: bool = True
    use_hardware_accel: bool = True
    use_bloom_filter: bool = True
    use_persistent_cache: bool = True
    use_intelligent_scan: bool = True
    parallel_workers: int = None
    batch_size: int = 10000
    
    @classmethod
    def auto_detect(cls) -> 'OptimizationStrategy':
        """Auto-detect optimal strategy based on system capabilities."""
        strategy = cls()
        
        # Detect CPU cores
        cpu_count = mp.cpu_count()
        strategy.parallel_workers = min(32, cpu_count * 2)
        
        # Detect available memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            # Adjust batch size based on available memory
            if available_memory > 8 * 1024 * 1024 * 1024:  # > 8GB
                strategy.batch_size = 50000
            elif available_memory > 4 * 1024 * 1024 * 1024:  # > 4GB
                strategy.batch_size = 20000
            else:
                strategy.batch_size = 5000
        except ImportError:
            pass
        
        return strategy


class OptimizedDuplicateAnalyzer:
    """High-performance duplicate analyzer with all optimizations integrated."""
    
    def __init__(self, config: Config = None, strategy: OptimizationStrategy = None):
        """Initialize optimized analyzer."""
        self.config = config or Config()
        self.strategy = strategy or OptimizationStrategy.auto_detect()
        
        # Initialize optimized components
        self.scanner = IntelligentScanner(self.config) if self.strategy.use_intelligent_scan else None
        self.async_scanner = AsyncOptimizedScanner(self.config) if self.strategy.use_async_io else None
        self.hasher = ParallelOptimizedHasher(self.config, self.strategy.parallel_workers)
        self.database = OptimizedDatabase(self.config)
        self.memory_index = MemoryIndex()
        
        # Initialize database
        self.database.initialize_optimized()
        
        # Statistics
        self.stats = {
            'scan_time': 0.0,
            'hash_time': 0.0,
            'db_time': 0.0,
            'total_time': 0.0,
            'files_processed': 0,
            'duplicates_found': 0,
            'space_wasted': 0
        }
    
    async def analyze_async(self, paths: List[Path]) -> AnalysisResult:
        """Analyze paths asynchronously for maximum performance."""
        start_time = time.time()
        
        # Phase 1: Async scanning with caching
        logger.info("Phase 1: Scanning file system...")
        scan_start = time.time()
        
        all_files = []
        if self.async_scanner and self.strategy.use_async_io:
            for path in paths:
                folder = await self.async_scanner.scan_tree_async(path)
                all_files.extend(self._extract_files(folder))
        else:
            for path in paths:
                folder = self.scanner.scan_with_inode_detection(path)
                all_files.extend(self._extract_files(folder))
        
        self.stats['scan_time'] = time.time() - scan_start
        self.stats['files_processed'] = len(all_files)
        logger.info(f"Scanned {len(all_files)} files in {self.stats['scan_time']:.2f}s")
        
        # Phase 2: Size-based grouping and early filtering
        logger.info("Phase 2: Grouping by size and filtering unique files...")
        size_groups = self._group_by_size(all_files)
        
        # Filter out unique sizes (no duplicates possible)
        potential_duplicates = []
        for size, files in size_groups.items():
            if len(files) > 1:
                potential_duplicates.extend(files)
                # Add to memory index for fast lookup
                for file in files:
                    self.memory_index.add_file(file)
        
        logger.info(f"Found {len(potential_duplicates)} potential duplicates to hash")
        
        # Phase 3: Intelligent hashing with multiple strategies
        logger.info("Phase 3: Computing file hashes...")
        hash_start = time.time()
        
        duplicate_groups = await self._hash_and_group_async(size_groups)
        
        self.stats['hash_time'] = time.time() - hash_start
        logger.info(f"Hashing completed in {self.stats['hash_time']:.2f}s")
        
        # Phase 4: Persist to database with bulk operations
        logger.info("Phase 4: Persisting results to database...")
        db_start = time.time()
        
        # Bulk insert files
        self.database.bulk_insert_files(all_files)
        
        # Bulk update hashes
        hash_updates = [(str(f.path), f.hash) for f in potential_duplicates if f.hash]
        self.database.bulk_update_hashes(hash_updates)
        
        self.stats['db_time'] = time.time() - db_start
        
        # Phase 5: Generate results
        self.stats['total_time'] = time.time() - start_time
        self.stats['duplicates_found'] = sum(len(g.files) - 1 for g in duplicate_groups)
        self.stats['space_wasted'] = sum((len(g.files) - 1) * g.size for g in duplicate_groups)
        
        # Save caches
        self.hasher.save_cache()
        self.scanner.save_cache()
        
        return AnalysisResult(
            total_files=len(all_files),
            duplicate_groups=duplicate_groups,
            total_size=sum(f.size for f in all_files),
            duplicate_size=self.stats['space_wasted'],
            scan_duration=self.stats['total_time']
        )
    
    def analyze(self, paths: List[Path]) -> AnalysisResult:
        """Synchronous analyze method."""
        # Run async method in event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.analyze_async(paths))
        finally:
            loop.close()
    
    async def _hash_and_group_async(self, size_groups: Dict[int, List[FileInfo]]) -> List[DuplicateGroup]:
        """Hash files and group duplicates asynchronously."""
        duplicate_groups = []
        
        # Progress tracking
        total_groups = len(size_groups)
        with tqdm(total=total_groups, desc="Processing size groups") as pbar:
            
            # Process size groups with different strategies based on size
            for size, files in size_groups.items():
                if len(files) <= 1:
                    continue
                
                # Choose hashing strategy based on file characteristics
                if size < 1024 * 1024:  # Small files < 1MB
                    # Hash all files completely
                    hash_results = self.hasher.parallel_hash_files([f.path for f in files])
                    
                elif size < 100 * 1024 * 1024:  # Medium files < 100MB
                    # Use quick hash first, then full hash for matches
                    quick_groups = await self._quick_hash_filter(files)
                    hash_results = {}
                    
                    for quick_hash, quick_files in quick_groups.items():
                        if len(quick_files) > 1:
                            full_hashes = self.hasher.parallel_hash_files([f.path for f in quick_files])
                            hash_results.update(full_hashes)
                    
                else:  # Large files >= 100MB
                    # Use sampling strategy
                    hash_results = {}
                    for file in files:
                        hash_value = self.hasher.hash_file_sampling(file.path)
                        if hash_value:
                            hash_results[str(file.path)] = hash_value
                
                # Update file objects with hashes
                for file in files:
                    if str(file.path) in hash_results:
                        file.hash = hash_results[str(file.path)]
                
                # Group by hash
                hash_groups = defaultdict(list)
                for file in files:
                    if file.hash:
                        hash_groups[file.hash].append(file)
                
                # Create duplicate groups
                for hash_value, hash_files in hash_groups.items():
                    if len(hash_files) > 1:
                        duplicate_groups.append(DuplicateGroup(
                            hash=hash_value,
                            size=size,
                            files=hash_files
                        ))
                
                pbar.update(1)
        
        return duplicate_groups
    
    async def _quick_hash_filter(self, files: List[FileInfo]) -> Dict[str, List[FileInfo]]:
        """Group files by quick hash for initial filtering."""
        quick_groups = defaultdict(list)
        
        # Compute quick hashes in parallel
        quick_hashes = self.hasher.batch_hash_files(
            [f.path for f in files],
            use_sampling=False  # Use quick hash instead
        )
        
        for file in files:
            quick_hash = quick_hashes.get(str(file.path))
            if quick_hash:
                file.quick_hash = quick_hash
                quick_groups[quick_hash].append(file)
        
        return quick_groups
    
    def _group_by_size(self, files: List[FileInfo]) -> Dict[int, List[FileInfo]]:
        """Group files by size."""
        size_groups = defaultdict(list)
        for file in files:
            size_groups[file.size].append(file)
        return size_groups
    
    def _extract_files(self, folder: FolderInfo) -> List[FileInfo]:
        """Extract all files from folder hierarchy."""
        files = list(folder.files)
        for subfolder in folder.subfolders:
            files.extend(self._extract_files(subfolder))
        return files
    
    def get_performance_report(self) -> Dict:
        """Generate detailed performance report."""
        report = {
            'timing': {
                'scan_time': f"{self.stats['scan_time']:.2f}s",
                'hash_time': f"{self.stats['hash_time']:.2f}s",
                'db_time': f"{self.stats['db_time']:.2f}s",
                'total_time': f"{self.stats['total_time']:.2f}s"
            },
            'throughput': {
                'files_per_second': self.stats['files_processed'] / max(self.stats['total_time'], 0.001),
                'mb_per_second': 0  # Calculate based on actual data processed
            },
            'efficiency': {
                'cache_hit_rate': f"{self.hasher.stats.cache_hits / max(self.hasher.stats.cache_hits + self.hasher.stats.cache_misses, 1) * 100:.1f}%",
                'mmap_usage': self.hasher.stats.mmap_used,
                'parallel_workers': self.strategy.parallel_workers
            },
            'results': {
                'files_processed': self.stats['files_processed'],
                'duplicates_found': self.stats['duplicates_found'],
                'space_wasted': f"{self.stats['space_wasted'] / (1024 * 1024 * 1024):.2f} GB"
            }
        }
        
        # Add database statistics
        report['database'] = self.database.create_duplicate_report()
        
        return report
    
    def optimize_for_ssd(self):
        """Optimize settings for SSD storage."""
        self.strategy.batch_size = 50000
        self.strategy.parallel_workers = mp.cpu_count() * 2
        self.hasher.MMAP_THRESHOLD = 1024 * 1024  # Use mmap more aggressively
    
    def optimize_for_hdd(self):
        """Optimize settings for HDD storage."""
        self.strategy.batch_size = 5000
        self.strategy.parallel_workers = mp.cpu_count()
        self.hasher.CHUNK_SIZE_MAP[float('inf')] = 512 * 1024  # Larger chunks for sequential reads
    
    def optimize_for_network_share(self):
        """Optimize settings for network shares."""
        self.strategy.use_async_io = True
        self.strategy.batch_size = 1000
        self.strategy.parallel_workers = min(4, mp.cpu_count())
        self.hasher.MMAP_THRESHOLD = 100 * 1024 * 1024  # Avoid mmap for network files
    
    def cleanup(self):
        """Clean up resources."""
        self.hasher.save_cache()
        self.scanner.save_cache()
        self.database.vacuum_database()
        self.database.close()


def create_optimized_analyzer(paths: List[Path], storage_type: str = 'auto') -> OptimizedDuplicateAnalyzer:
    """Factory function to create optimized analyzer with auto-detection."""
    config = Config()
    strategy = OptimizationStrategy.auto_detect()
    
    analyzer = OptimizedDuplicateAnalyzer(config, strategy)
    
    # Apply storage-specific optimizations
    if storage_type == 'ssd':
        analyzer.optimize_for_ssd()
    elif storage_type == 'hdd':
        analyzer.optimize_for_hdd()
    elif storage_type == 'network':
        analyzer.optimize_for_network_share()
    # 'auto' uses default settings
    
    return analyzer