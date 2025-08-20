"""Command-line interface for PyDeduper."""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
import json

from .config import Config, HashAlgorithm
from .core.scanner import ProgressScanner
from .core.analyzer import DuplicateAnalyzer
from .storage.database import Database
from .storage.models import AnalysisResult


def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration."""
    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def format_size(size: int) -> str:
    """Format size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def print_progress(current: int, path: str):
    """Print progress during scanning."""
    print(f"\rScanning... {current} files processed. Current: {path[:50]}...", end="", flush=True)


def print_duplicate_groups(result: AnalysisResult, show_all: bool = False):
    """Print duplicate file groups."""
    if not result.duplicate_groups:
        print("\nNo duplicate files found.")
        return
    
    print(f"\nFound {len(result.duplicate_groups)} groups of duplicate files")
    print(f"Total duplicates: {result.total_duplicates} files")
    print(f"Total wasted space: {format_size(result.total_wasted_space)}\n")
    
    # Sort groups by wasted space
    sorted_groups = sorted(result.duplicate_groups, 
                          key=lambda g: g.total_wasted_space, 
                          reverse=True)
    
    # Show top 10 unless --all is specified
    groups_to_show = sorted_groups if show_all else sorted_groups[:10]
    
    for i, group in enumerate(groups_to_show, 1):
        print(f"Group {i}: {len(group.files)} files, {format_size(group.size)} each")
        print(f"  Hash: {group.hash}")
        print(f"  Wasted space: {format_size(group.total_wasted_space)}")
        print("  Files:")
        
        for file in group.files[:5]:  # Show max 5 files per group
            print(f"    - {file.full_path}")
        
        if len(group.files) > 5:
            print(f"    ... and {len(group.files) - 5} more")
        print()
    
    if not show_all and len(sorted_groups) > 10:
        print(f"(Showing top 10 groups. Use --all to see all {len(sorted_groups)} groups)")


def print_folder_stats(result: AnalysisResult, threshold: float = 0.0):
    """Print folder duplication statistics."""
    if not result.folder_duplication_stats:
        return
    
    print("\nFolder Duplication Statistics:")
    print("-" * 50)
    
    # Sort by duplication percentage
    sorted_stats = sorted(result.folder_duplication_stats.items(), 
                         key=lambda x: x[1], 
                         reverse=True)
    
    for folder_path, dup_pct in sorted_stats:
        if dup_pct >= threshold:
            print(f"{dup_pct:6.2f}% - {folder_path}")


def export_results(result: AnalysisResult, output_file: Path, format: str = "json"):
    """Export analysis results to file."""
    if format == "json":
        data = {
            "total_duplicates": result.total_duplicates,
            "total_wasted_space": result.total_wasted_space,
            "duplicate_groups": [
                {
                    "hash": group.hash,
                    "size": group.size,
                    "count": group.count,
                    "files": [f.full_path for f in group.files]
                }
                for group in result.duplicate_groups
            ],
            "folder_stats": result.folder_duplication_stats
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format == "csv":
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Hash", "Size", "Count", "Files"])
            
            for group in result.duplicate_groups:
                files_str = "|".join(f.full_path for f in group.files)
                writer.writerow([group.hash, group.size, group.count, files_str])
    
    print(f"Results exported to {output_file}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="PyDeduper - Find duplicate files with folder analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional arguments
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Paths to scan for duplicates"
    )
    
    # Optional arguments
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
        "--min-size",
        type=int,
        default=0,
        help="Minimum file size to consider (in bytes)"
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
    
    if args.db:
        config.DEFAULT_DB_PATH = args.db
    
    try:
        # Validate paths
        for path in args.paths:
            if not path.exists():
                print(f"Error: Path does not exist: {path}", file=sys.stderr)
                return 1
        
        # Create analyzer
        analyzer = DuplicateAnalyzer(config)
        
        # Perform analysis
        if args.no_db:
            print("Analyzing files in memory...")
            scanner = ProgressScanner(config, print_progress if args.verbose else None)
            folders = scanner.scan_multiple(args.paths)
            
            if len(folders) == 1:
                result = analyzer.analyze_folder(folders[0])
            else:
                # Combine multiple folders analysis
                print("Multiple path analysis not yet fully implemented for in-memory mode")
                return 1
        else:
            print("Analyzing files using database...")
            result = analyzer.analyze_with_database(args.paths)
        
        print()  # Clear progress line
        
        # Find similar folders if requested
        if args.find_similar is not None:
            print(f"\nFinding folders with >= {args.find_similar}% similarity...")
            similar = analyzer.find_similar_folders(args.find_similar)
            
            if similar:
                print(f"Found {len(similar)} similar folder pairs:\n")
                for folder1, folder2, similarity in similar[:10]:
                    print(f"  {similarity:.1f}% similar:")
                    print(f"    - {folder1}")
                    print(f"    - {folder2}")
                    print()
            else:
                print("No similar folders found.")
        
        # Print results
        print_duplicate_groups(result, args.all)
        print_folder_stats(result, args.folder_threshold)
        
        # Export if requested
        if args.export:
            export_results(result, args.export, args.export_format)
        
        # Print summary
        stats = analyzer.db.get_folder_statistics() if not args.no_db else {}
        if stats:
            print("\nOverall Statistics:")
            print(f"  Total folders: {stats['total_folders']}")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total size: {format_size(stats['total_size'])}")
            print(f"  Unique files: {stats['unique_hashes']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if not args.no_db:
            analyzer.db.close()


if __name__ == "__main__":
    sys.exit(main())