"""Enhanced analyzer with parallel processing and advanced progress reporting."""

import logging
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Set, Callable, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys

try:
    from tqdm import tqdm
except ImportError:
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

from ..config import Config
from ..storage.models import (
    FileInfo, FolderInfo, DuplicateGroup, 
    AnalysisResult, ScanResult
)
from ..storage.database import Database
from .analyzer import DuplicateAnalyzer
from .parallel_hasher import ParallelHasher, AdaptiveHasher
from .parallel_scanner import ParallelScanner, create_parallel_scanner

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Advanced progress tracking with multiple phases and ETA calculation."""
    
    def __init__(self, total_phases: int, phase_names: List[str] = None):
        """
        Initialize progress tracker.
        
        Args:
            total_phases: Number of analysis phases
            phase_names: Optional names for each phase
        """
        self.total_phases = total_phases
        self.phase_names = phase_names or [f"Phase {i+1}" for i in range(total_phases)]
        self.current_phase = 0
        self.phase_progress = {}
        self.start_time = time.time()
        self.phase_start_times = {}
        self.cancelled = threading.Event()
        
        # Main progress bar
        self.main_pbar = tqdm(
            total=total_phases,
            desc="Analysis Progress",
            unit='phases',
            position=0
        )
        
        # Current phase progress bar
        self.phase_pbar = None
    
    def start_phase(self, phase_index: int, total_items: int, description: str = None):
        """Start a new analysis phase."""
        if self.phase_pbar:
            self.phase_pbar.close()
        
        self.current_phase = phase_index
        self.phase_start_times[phase_index] = time.time()
        
        phase_name = self.phase_names[phase_index] if phase_index < len(self.phase_names) else f"Phase {phase_index + 1}"
        desc = description or phase_name
        
        self.phase_pbar = tqdm(
            total=total_items,
            desc=desc,
            unit='items',
            position=1,
            leave=False
        )
        
        self.phase_progress[phase_index] = {
            'total': total_items,
            'completed': 0,
            'start_time': time.time()
        }
    
    def update_phase(self, increment: int = 1, status: str = None):
        """Update current phase progress."""
        if not self.phase_pbar or self.current_phase not in self.phase_progress:
            return
        
        self.phase_progress[self.current_phase]['completed'] += increment
        self.phase_pbar.update(increment)
        
        if status:
            # Calculate ETA for current phase
            progress_info = self.phase_progress[self.current_phase]
            elapsed = time.time() - progress_info['start_time']
            completed = progress_info['completed']
            total = progress_info['total']
            
            if completed > 0:
                rate = completed / elapsed
                remaining = total - completed
                eta = remaining / rate if rate > 0 else 0
                self.phase_pbar.set_description(f"{status} (ETA: {eta:.0f}s)")
    
    def finish_phase(self):
        """Finish the current phase."""
        if self.phase_pbar:
            self.phase_pbar.close()
            self.phase_pbar = None
        
        self.main_pbar.update(1)
        
        # Update main progress bar description with overall ETA
        overall_elapsed = time.time() - self.start_time
        completed_phases = self.current_phase + 1
        remaining_phases = self.total_phases - completed_phases
        
        if completed_phases > 0:
            avg_phase_time = overall_elapsed / completed_phases
            overall_eta = remaining_phases * avg_phase_time
            self.main_pbar.set_description(f"Analysis Progress (ETA: {overall_eta:.0f}s)")
    
    def cancel(self):
        """Cancel the analysis."""
        self.cancelled.set()
        if self.phase_pbar:
            self.phase_pbar.close()
        self.main_pbar.close()
    
    def is_cancelled(self) -> bool:
        """Check if analysis was cancelled."""
        return self.cancelled.is_set()
    
    def close(self):
        """Close all progress bars."""
        if self.phase_pbar:
            self.phase_pbar.close()
        self.main_pbar.close()


class CancellationHandler:
    """Handles graceful cancellation of long-running operations."""
    
    def __init__(self):
        self.cancelled = threading.Event()
        self.cleanup_functions = []
        
        # Register signal handlers for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle cancellation signals."""
        print("\n\nCancellation requested... cleaning up...")
        self.cancel()
    
    def cancel(self):
        """Request cancellation."""
        self.cancelled.set()
        
        # Run cleanup functions
        for cleanup_fn in self.cleanup_functions:
            try:
                cleanup_fn()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self.cancelled.is_set()
    
    def add_cleanup(self, cleanup_fn: Callable):
        """Add a cleanup function to be called on cancellation."""
        self.cleanup_functions.append(cleanup_fn)


class ParallelDuplicateAnalyzer(DuplicateAnalyzer):
    """Enhanced duplicate analyzer with parallel processing and progress reporting."""
    
    def __init__(self, config: Config = None, max_workers: int = None, 
                 enable_progress: bool = True):
        """
        Initialize parallel analyzer.
        
        Args:
            config: Configuration object
            max_workers: Maximum number of worker threads
            enable_progress: Whether to show progress bars
        """
        super().__init__(config)
        self.max_workers = max_workers
        self.enable_progress = enable_progress
        self.parallel_hasher = AdaptiveHasher(config, max_workers)
        self.cancellation_handler = CancellationHandler()
        
        # Add cleanup handlers
        self.cancellation_handler.add_cleanup(self.parallel_hasher.cancel)
        self.cancellation_handler.add_cleanup(self.db.close)
    
    def is_cancelled(self) -> bool:
        """Check if analysis was cancelled."""
        return self.cancellation_handler.is_cancelled()
    
    def analyze_with_database_parallel(self, paths: List[Path]) -> AnalysisResult:
        """
        Analyze files using database with parallel processing and progress reporting.
        
        Args:
            paths: List of paths to analyze
            
        Returns:
            AnalysisResult with duplicate information
        """
        if not self.enable_progress:
            return super().analyze_with_database(paths)
        
        # Initialize progress tracker
        phase_names = [
            "Initialization",
            "File Scanning", 
            "Size Analysis",
            "Hash Calculation",
            "Duplicate Detection",
            "Statistics Calculation"
        ]
        
        progress = ProgressTracker(len(phase_names), phase_names)
        
        try:
            # Phase 1: Initialization
            progress.start_phase(0, 3, "Initializing database")
            
            self.db.initialize()
            progress.update_phase(1, "Database initialized")
            
            if self.is_cancelled():
                return AnalysisResult()
            
            if self.config.VERBOSE:
                logger.info("Clearing existing database...")
            self.db.clear_all()
            progress.update_phase(1, "Database cleared")
            
            progress.update_phase(1, "Initialization complete")
            progress.finish_phase()
            
            # Phase 2: File Scanning
            if self.is_cancelled():
                return AnalysisResult()
            
            progress.start_phase(1, len(paths), "Scanning file system")
            
            scanner = create_parallel_scanner(self.config, self.max_workers, False)
            folders = []
            
            for i, path in enumerate(paths):
                if self.is_cancelled():
                    return AnalysisResult()
                
                if path.is_file():
                    # Single file
                    file_info = scanner.scan_file(path)
                    if file_info:
                        folder = FolderInfo(
                            path=path.parent,
                            name=path.parent.name,
                            files=[file_info],
                            num_files=1,
                            total_size=file_info.size
                        )
                        folders.append(folder)
                elif path.is_dir():
                    # Directory tree
                    folder = scanner.scan_tree_parallel(path)
                    folders.append(folder)
                
                progress.update_phase(1, f"Scanned {path}")
            
            progress.finish_phase()
            
            # Phase 3: Database Storage
            if self.is_cancelled():
                return AnalysisResult()
            
            total_folders = sum(self._count_folders_recursive(folder) for folder in folders)
            progress.start_phase(2, total_folders, "Storing in database")
            
            for folder in folders:
                if self.is_cancelled():
                    return AnalysisResult()
                self._store_folder_recursive_with_progress(folder, progress)
            
            progress.finish_phase()
            
            # Phase 4: Size Analysis and Hash Calculation
            if self.is_cancelled():
                return AnalysisResult()
            
            duplicate_sizes = self.db.get_duplicate_sizes()
            total_files_to_hash = sum(count for size, count in duplicate_sizes)
            
            progress.start_phase(3, total_files_to_hash, "Calculating hashes")
            
            self._calculate_hashes_with_progress(duplicate_sizes, progress)
            progress.finish_phase()
            
            # Phase 5: Duplicate Detection
            if self.is_cancelled():
                return AnalysisResult()
            
            duplicate_hashes = self.db.get_duplicate_hashes()
            progress.start_phase(4, len(duplicate_hashes), "Building duplicate groups")
            
            result = AnalysisResult()
            for hash_val, size, count in duplicate_hashes:
                if self.is_cancelled():
                    return result
                
                files_data = self.db.get_files_by_hash(hash_val)
                files = [FileInfo.from_dict(f) for f in files_data]
                
                group = DuplicateGroup(
                    hash=hash_val,
                    size=size,
                    count=count,
                    files=files
                )
                result.duplicate_groups.append(group)
                
                progress.update_phase(1, f"Processed hash group {len(result.duplicate_groups)}")
            
            progress.finish_phase()
            
            # Phase 6: Statistics Calculation
            if self.is_cancelled():
                return result
            
            folder_count = len(list(self.db.get_connection().cursor().execute("SELECT COUNT(*) FROM folders").fetchone()))
            progress.start_phase(5, folder_count, "Calculating statistics")
            
            result.calculate_stats()
            result.folder_duplication_stats = self._calculate_db_folder_stats_with_progress(progress)
            
            progress.finish_phase()
            
            return result
        
        except KeyboardInterrupt:
            logger.info("Analysis cancelled by user")
            return AnalysisResult()
        
        finally:
            progress.close()
    
    def _count_folders_recursive(self, folder: FolderInfo) -> int:
        """Count total folders recursively."""
        count = 1  # Current folder
        for subfolder in folder.subfolders:
            count += self._count_folders_recursive(subfolder)
        return count
    
    def _store_folder_recursive_with_progress(self, folder: FolderInfo, progress: ProgressTracker):
        """Store folder recursively with progress updates."""
        if self.is_cancelled():
            return
        
        # Store folder
        folder_id = self.db.insert_folder(folder)
        
        # Store files
        if folder.files:
            for file in folder.files:
                file.parent_folder_id = folder_id
            self.db.batch_insert_files(folder.files, folder_id)
        
        progress.update_phase(1, f"Stored {folder.name}")
        
        # Recurse into subfolders
        for subfolder in folder.subfolders:
            if self.is_cancelled():
                return
            self._store_folder_recursive_with_progress(subfolder, progress)
    
    def _calculate_hashes_with_progress(self, duplicate_sizes: List[Tuple[int, int]], 
                                       progress: ProgressTracker):
        """Calculate hashes with progress reporting."""
        files_processed = 0
        
        for size, count in duplicate_sizes:
            if self.is_cancelled():
                return
            
            files_data = self.db.get_files_by_size(size)
            files_to_hash = []
            
            for file_data in files_data:
                if not file_data['hash']:
                    file_info = FileInfo(
                        id=file_data['id'],
                        path=Path(file_data['path']),
                        name=file_data['name'],
                        size=file_data['size']
                    )
                    files_to_hash.append(file_info)
            
            if files_to_hash:
                # Use parallel hasher
                def hash_progress_callback(completed, total, current_file):
                    progress.update_phase(1, f"Hashing {Path(current_file).name}")
                
                hash_results = self.parallel_hasher.calculate_hashes_parallel(
                    files_to_hash, 'full', hash_progress_callback
                )
                
                # Update database with results
                for file_info in files_to_hash:
                    hash_value = hash_results.get(str(file_info.path))
                    if hash_value and file_info.id:
                        self.db.update_file_hash(file_info.id, hash_value)
                
                files_processed += len(files_to_hash)
    
    def _calculate_db_folder_stats_with_progress(self, progress: ProgressTracker) -> Dict[str, float]:
        """Calculate folder duplication statistics from database with progress."""
        stats = {}
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, path FROM folders')
            folders = cursor.fetchall()
            
            for row in folders:
                if self.is_cancelled():
                    break
                
                folder_id = row['id']
                folder_path = row['path']
                duplication_pct = self.db.calculate_folder_duplication(folder_id)
                stats[folder_path] = duplication_pct
                
                progress.update_phase(1, f"Analyzed {Path(folder_path).name}")
        
        return stats
    
    def find_similar_folders_parallel(self, threshold: float = 70.0, 
                                     progress_callback: Optional[Callable] = None) -> List[Tuple[str, str, float]]:
        """
        Find similar folders using parallel processing.
        
        Args:
            threshold: Minimum similarity percentage
            progress_callback: Optional progress callback
            
        Returns:
            List of (folder1, folder2, similarity_percentage) tuples
        """
        similar_pairs = []
        
        # Get all folders with their file hashes
        folder_hashes = self._get_folder_hashes_parallel()
        folder_paths = list(folder_hashes.keys())
        
        # Calculate total comparisons needed
        total_comparisons = len(folder_paths) * (len(folder_paths) - 1) // 2
        
        if self.enable_progress:
            pbar = tqdm(total=total_comparisons, desc="Finding similar folders", unit='comparisons')
        
        try:
            # Use parallel processing for similarity calculations
            def compare_folders(args):
                i, j = args
                path1, path2 = folder_paths[i], folder_paths[j]
                hashes1, hashes2 = folder_hashes[path1], folder_hashes[path2]
                
                if hashes1 and hashes2:
                    similarity = self._calculate_similarity(hashes1, hashes2)
                    if similarity >= threshold:
                        return (path1, path2, similarity)
                return None
            
            # Generate comparison pairs
            comparison_pairs = [(i, j) for i in range(len(folder_paths)) 
                              for j in range(i + 1, len(folder_paths))]
            
            # Process comparisons in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pair = {executor.submit(compare_folders, pair): pair 
                                for pair in comparison_pairs}
                
                for future in as_completed(future_to_pair):
                    if self.is_cancelled():
                        break
                    
                    result = future.result()
                    if result:
                        similar_pairs.append(result)
                    
                    if self.enable_progress:
                        pbar.update(1)
        
        finally:
            if self.enable_progress:
                pbar.close()
        
        return sorted(similar_pairs, key=lambda x: x[2], reverse=True)
    
    def _get_folder_hashes_parallel(self) -> Dict[str, Set[str]]:
        """Get folder hashes using parallel database queries."""
        folder_hashes = defaultdict(set)
        
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
                folder_hashes[folder_path].add(file_hash)
        
        return dict(folder_hashes)
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        stats = {
            'hasher_stats': self.parallel_hasher.get_statistics(),
            'database_stats': self.db.get_folder_statistics(),
        }
        return stats