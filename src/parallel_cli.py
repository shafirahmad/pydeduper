"""Enhanced command-line interface with parallel processing and progress reporting."""

import argparse
import sys
import signal
import logging
from pathlib import Path
from typing import List, Optional
import json
import time
import os

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .config import Config, HashAlgorithm
from .core.parallel_scanner import create_parallel_scanner
from .core.parallel_analyzer import ParallelDuplicateAnalyzer
from .storage.parallel_database import ParallelDatabase
from .storage.models import AnalysisResult
from .cli import (
    setup_logging, format_size, print_duplicate_groups, 
    print_folder_stats, export_results
)


class InteractiveProgressReporter:
    """Interactive progress reporter with user controls."""
    
    def __init__(self, enable_interactive: bool = True):
        self.enable_interactive = enable_interactive
        self.cancelled = False
        self.paused = False
        
        if enable_interactive:
            # Setup signal handlers for interactive control
            signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n⚠️  Interruption detected!")
        print("Options:")
        print("  c - Cancel operation")
        print("  r - Resume operation")
        print("  s - Show current statistics")
        print("  q - Quit immediately")
        
        while True:
            try:
                choice = input("\nEnter choice (c/r/s/q): ").lower().strip()
                if choice == 'c':
                    print("🛑 Cancelling operation...")
                    self.cancelled = True
                    break
                elif choice == 'r':
                    print("▶️  Resuming operation...")
                    break
                elif choice == 's':
                    self._show_statistics()
                elif choice == 'q':
                    print("🚪 Quitting immediately...")
                    sys.exit(130)
                else:
                    print("Invalid choice. Please enter c, r, s, or q.")
            except (EOFError, KeyboardInterrupt):
                print("\n🚪 Quitting...")
                sys.exit(130)
    
    def _show_statistics(self):
        """Show current operation statistics."""
        # This would be populated by the actual operation
        print("\n📊 Current Statistics:")
        print("  Files processed: [would show current count]")
        print("  Current phase: [would show current phase]")
        print("  Elapsed time: [would show elapsed time]")
        print("  Estimated remaining: [would show ETA]")
    
    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self.cancelled


def print_performance_stats(analyzer: ParallelDuplicateAnalyzer):
    """Print performance statistics."""
    stats = analyzer.get_performance_stats()
    
    print("\n📊 Performance Statistics:")
    print("=" * 50)
    
    # Hasher statistics
    hasher_stats = stats.get('hasher_stats', {})
    if hasher_stats:
        print(f"Files processed: {hasher_stats.get('files_processed', 0):,}")
        print(f"Cache hit rate: {hasher_stats.get('cache_hit_rate', 0):.1%}")
        print(f"Average time per file: {hasher_stats.get('avg_time_per_file', 0):.3f}s")
        print(f"Hash calculation errors: {hasher_stats.get('errors', 0)}")
    
    # Database statistics
    db_stats = stats.get('database_stats', {})
    if db_stats:
        print(f"Total folders: {db_stats.get('total_folders', 0):,}")
        print(f"Total files: {db_stats.get('total_files', 0):,}")
        print(f"Total size: {format_size(db_stats.get('total_size', 0))}")
        print(f"Unique files: {db_stats.get('unique_hashes', 0):,}")


def print_analysis_summary(result: AnalysisResult, start_time: float):
    """Print comprehensive analysis summary."""
    elapsed_time = time.time() - start_time
    
    print("\n🎯 Analysis Summary:")
    print("=" * 50)
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"📁 Duplicate groups: {len(result.duplicate_groups):,}")
    print(f"📄 Duplicate files: {result.total_duplicates:,}")
    print(f"💾 Wasted space: {format_size(result.total_wasted_space)}")
    
    if result.duplicate_groups:
        # Show top wasted space
        top_group = max(result.duplicate_groups, key=lambda g: g.total_wasted_space)
        print(f"🏆 Largest waste: {format_size(top_group.total_wasted_space)} ({top_group.count} copies of {format_size(top_group.size)})")
        
        # Show most duplicated
        most_duped = max(result.duplicate_groups, key=lambda g: g.count)
        print(f"📈 Most copies: {most_duped.count} copies of {format_size(most_duped.size)} file")


def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create argument parser with enhanced parallel options."""
    parser = argparse.ArgumentParser(
        description="PyDeduper - Parallel duplicate file finder with advanced progress reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic scan with progress
  %(prog)s /path/to/directory

  # Parallel scan with 8 workers
  %(prog)s --workers 8 /path/to/directory

  # High-performance scan with optimizations
  %(prog)s --fast --workers 16 --no-progress /path/to/directory

  # Interactive mode with cancellation support
  %(prog)s --interactive /path/to/directory

  # Memory-efficient scan for large datasets
  %(prog)s --batch-size 500 --max-connections 4 /path/to/directory
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Paths to scan for duplicates"
    )
    
    # Basic options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    # Performance options
    parser.add_argument(
        "--workers", "-w",
        type=int,
        help="Number of parallel workers (default: auto-detect)"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable all performance optimizations"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for database operations (default: 1000)"
    )
    
    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help="Maximum database connections (default: 10)"
    )
    
    # Progress and interaction options
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enable interactive mode with cancellation support"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed performance statistics"
    )
    
    # Original CLI options (inherited)
    parser.add_argument(
        "--hash",
        choices=["md5", "sha1", "sha256", "sha512"],
        default="sha256",
        help="Hash algorithm to use (default: sha256)"
    )
    
    parser.add_argument(
        "--db",
        type=Path,
        help="Database file path (default: data/pydeduper.db)"
    )
    
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Don't use database, analyze in memory only"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all duplicate groups (not just top 10)"
    )
    
    parser.add_argument(
        "--export",
        type=Path,
        help="Export results to file"
    )
    
    parser.add_argument(
        "--export-format",
        choices=["json", "csv"],
        default="json",
        help="Export format (default: json)"
    )
    
    parser.add_argument(
        "--folder-threshold",
        type=float,
        default=0.0,
        help="Minimum duplication percentage to show folders"
    )
    
    parser.add_argument(
        "--find-similar",
        type=float,
        metavar="THRESHOLD",
        help="Find folders with similarity >= THRESHOLD%%"
    )
    
    parser.add_argument(
        "--ignore-hidden",
        action="store_true",
        help="Ignore hidden files and folders"
    )
    
    parser.add_argument(
        "--follow-links",
        action="store_true",
        help="Follow symbolic links"
    )
    
    return parser


def main():
    """Enhanced main function with parallel processing."""
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.debug)
    
    # Create configuration
    config = Config()
    config.VERBOSE = args.verbose
    config.DEBUG = args.debug
    config.DEFAULT_HASH_ALGORITHM = HashAlgorithm(args.hash)
    config.IGNORE_HIDDEN = args.ignore_hidden
    config.FOLLOW_SYMLINKS = args.follow_links
    config.BATCH_SIZE = args.batch_size
    
    if args.db:
        config.DEFAULT_DB_PATH = args.db
    
    # Apply fast mode optimizations
    if args.fast:
        if not args.workers:
            args.workers = min(32, (os.cpu_count() or 1) * 2)
        args.no_progress = True  # Disable progress for maximum speed
        config.BATCH_SIZE = 2000  # Larger batches
    
    # Determine worker count
    max_workers = args.workers or min(32, (os.cpu_count() or 1) + 4)
    
    # Setup interactive reporter
    interactive = InteractiveProgressReporter(args.interactive)
    
    try:
        # Validate paths
        for path in args.paths:
            if not path.exists():
                print(f"❌ Error: Path does not exist: {path}", file=sys.stderr)
                return 1
        
        # Check for tqdm if progress is enabled
        if not args.no_progress and not TQDM_AVAILABLE:
            print("⚠️  Warning: tqdm not available, progress bars disabled")
            print("   Install with: pip install tqdm")
            args.no_progress = True
        
        start_time = time.time()
        
        # Create analyzer with parallel processing
        if args.no_db:
            print("🔄 Analyzing files in memory...")
            # Note: In-memory analysis with parallel processing would need additional implementation
            print("❌ Parallel in-memory analysis not yet implemented")
            print("   Use database mode for parallel processing")
            return 1
        else:
            print("🚀 Starting parallel analysis...")
            if not args.no_progress:
                print("💡 Tip: Use Ctrl+C for interactive controls")
            
            # Create parallel analyzer
            analyzer = ParallelDuplicateAnalyzer(
                config=config,
                max_workers=max_workers,
                enable_progress=not args.no_progress
            )
            
            # Override database with parallel version
            analyzer.db = ParallelDatabase(config, args.max_connections)
            
            # Perform analysis
            result = analyzer.analyze_with_database_parallel(args.paths)
        
        # Check if cancelled
        if interactive.is_cancelled():
            print("\n🛑 Analysis cancelled by user")
            return 130
        
        # Find similar folders if requested
        if args.find_similar is not None:
            print(f"\n🔍 Finding folders with >= {args.find_similar}% similarity...")
            
            def similarity_progress(current, total):
                if not args.no_progress:
                    print(f"\rProgress: {current}/{total} comparisons", end="", flush=True)
            
            similar = analyzer.find_similar_folders_parallel(
                args.find_similar, 
                similarity_progress if not args.no_progress else None
            )
            
            if not args.no_progress:
                print()  # New line
            
            if similar:
                print(f"✅ Found {len(similar)} similar folder pairs:\n")
                for folder1, folder2, similarity in similar[:10]:
                    print(f"  📊 {similarity:.1f}% similar:")
                    print(f"    📁 {folder1}")
                    print(f"    📁 {folder2}")
                    print()
            else:
                print("ℹ️  No similar folders found.")
        
        # Print results
        print_analysis_summary(result, start_time)
        print_duplicate_groups(result, args.all)
        print_folder_stats(result, args.folder_threshold)
        
        # Show performance statistics
        if args.stats:
            print_performance_stats(analyzer)
        
        # Export if requested
        if args.export:
            export_results(result, args.export, args.export_format)
        
        return 0
        
    except KeyboardInterrupt:
        if not interactive.is_cancelled():
            print("\n\n🛑 Operation interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Cleanup
        try:
            if 'analyzer' in locals():
                analyzer.db.close()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())