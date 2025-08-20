# PyDeduper Code Explanation

## Overview
PyDeduper is a duplicate file detection system with advanced folder analysis capabilities. The application is built using a modular architecture that separates concerns into distinct components for scanning, hashing, analysis, and storage.

## Architecture

### Core Components

```
src/
├── core/           # Core business logic
├── storage/        # Data persistence layer
├── cli.py          # Command-line interface
└── config.py       # Configuration management
```

## Module Breakdown

### 1. Configuration (`src/config.py`)

The configuration module centralizes all settings for the application:

```python
class Config:
    CHUNK_SIZE = 8192  # Size for reading files in chunks
    DEFAULT_HASH_ALGORITHM = HashAlgorithm.SHA256
    QUICK_HASH_SIZE = 1024  # Bytes to read for quick hash
    DEFAULT_DB_PATH = Path("data/pydeduper.db")
```

**Key Features:**
- Enum-based hash algorithm selection (MD5, SHA1, SHA256, SHA512)
- Environment variable support for runtime configuration
- Default settings with override capabilities
- Pattern-based file ignoring (temp files, caches, etc.)

### 2. Data Models (`src/storage/models.py`)

Defines the data structures used throughout the application:

#### FileInfo
```python
@dataclass
class FileInfo:
    path: Path
    name: str
    size: int
    hash: Optional[str]
    quick_hash: Optional[str]
    last_modified: Optional[datetime]
```
Represents individual file metadata including path, size, and hash values.

#### FolderInfo
```python
@dataclass
class FolderInfo:
    path: Path
    files: List[FileInfo]
    subfolders: List["FolderInfo"]
    duplication_percentage: float
```
Represents folder structure with recursive subfolder support and duplication statistics.

#### DuplicateGroup
```python
@dataclass
class DuplicateGroup:
    hash: str
    size: int
    count: int
    files: List[FileInfo]
    total_wasted_space: int
```
Groups duplicate files together with waste calculation.

### 3. File Hashing (`src/core/hasher.py`)

Handles all hash calculations with optimization strategies:

#### FileHasher Class
```python
class FileHasher:
    def hash_file(self, file_path: Path) -> Optional[str]:
        # Full file hash calculation
        
    def quick_hash(self, file_path: Path) -> Optional[str]:
        # Hash only first 1KB for quick comparison
        
    def hash_file_partial(self, file_path: Path) -> Optional[str]:
        # Sample-based hashing (beginning, middle, end)
```

**Optimization Strategy:**
1. **Quick Hash**: For initial filtering, only hash first 1KB
2. **Full Hash**: Calculate complete hash only for potential duplicates
3. **Partial Hash**: Sample strategic file positions for large files

#### HashCache Class
```python
class HashCache:
    def get_hash(self, file_path: Path, mtime: float) -> Optional[str]:
        # Returns cached hash if file hasn't changed
        
    def set_hash(self, file_path: Path, hash_value: str, mtime: float):
        # Stores hash with modification time validation
```

Implements an in-memory cache to avoid recalculating hashes for unchanged files.

### 4. File System Scanner (`src/core/scanner.py`)

Traverses the file system and collects file information:

#### FileScanner Class
```python
class FileScanner:
    def scan_directory(self, directory: Path) -> Generator[Tuple[Path, List[FileInfo]]]:
        # Yields (folder_path, files_in_folder) tuples
        
    def scan_tree(self, root_path: Path) -> FolderInfo:
        # Builds complete hierarchical folder structure
```

**Features:**
- Generator-based for memory efficiency
- Configurable recursion depth
- Pattern-based file ignoring
- Symlink handling options
- Error recovery with logging

#### ProgressScanner
Extends FileScanner with progress reporting callbacks for UI updates.

### 5. Duplicate Analysis (`src/core/analyzer.py`)

Core logic for finding and analyzing duplicates:

#### DuplicateAnalyzer Class
```python
class DuplicateAnalyzer:
    def find_duplicates_by_size(self, files: List[FileInfo]) -> Dict[int, List[FileInfo]]:
        # Step 1: Group files by size
        
    def calculate_hashes_for_duplicates(self, size_groups: Dict) -> Dict[str, List[FileInfo]]:
        # Step 2: Calculate hashes for same-sized files
        
    def analyze_folder(self, folder: FolderInfo) -> AnalysisResult:
        # Complete duplicate analysis for a folder tree
```

**Duplicate Detection Algorithm:**
1. **Size Grouping**: Files with unique sizes can't be duplicates
2. **Quick Hash Filter**: For large files, compare quick hashes first
3. **Full Hash Verification**: Calculate full hash only for quick hash matches
4. **Group Formation**: Create duplicate groups with statistics

#### Folder Analysis
```python
def calculate_folder_duplication(self, folder_id: int) -> float:
    # Percentage of files in folder that exist elsewhere
    
def find_similar_folders(self, threshold: float) -> List[Tuple[str, str, float]]:
    # Find folders with similar content based on file hashes
```

Calculates duplication percentages and finds similar folders using set operations on file hashes.

### 6. Database Layer (`src/storage/database.py`)

SQLite-based persistence with proper connection management:

#### Database Class
```python
class Database:
    @contextmanager
    def get_connection(self):
        # Context manager for safe connection handling
        
    def initialize(self):
        # Creates tables and indexes
        
    def batch_insert_files(self, files: List[FileInfo]):
        # Efficient bulk insertion
```

**Schema:**
- **folders** table: Stores folder metadata and hierarchy
- **files** table: Stores file information with foreign key to folders
- **Indexes**: On hash, size, and quick_hash for performance

**Key Operations:**
```sql
-- Find duplicate sizes
SELECT size, COUNT(*) as count
FROM files
GROUP BY size
HAVING count > 1

-- Find duplicate hashes
SELECT hash, size, COUNT(*) as count
FROM files
WHERE hash IS NOT NULL
GROUP BY hash
HAVING count > 1

-- Calculate folder duplication
SELECT COUNT(DISTINCT f1.id) as duplicated_count
FROM files f1
INNER JOIN files f2 ON f1.hash = f2.hash
WHERE f1.parent_folder_id = ?
  AND f2.id != f1.id
```

### 7. Command-Line Interface (`src/cli.py`)

User interface with argument parsing and result formatting:

```python
def main():
    parser = argparse.ArgumentParser()
    # Parse arguments for paths, options, export format
    
    analyzer = DuplicateAnalyzer(config)
    result = analyzer.analyze_with_database(args.paths)
    
    print_duplicate_groups(result)
    print_folder_stats(result)
    export_results(result, args.export)
```

**Features:**
- Multiple path support
- Various hash algorithms
- Export to JSON/CSV
- Similarity threshold configuration
- Progress reporting
- Debug/verbose modes

## Data Flow

1. **User Input** → CLI parses arguments and paths
2. **Scanning** → FileScanner traverses directories, collecting FileInfo objects
3. **Storage** → Database stores file and folder information
4. **Analysis** → DuplicateAnalyzer processes files:
   - Groups by size
   - Calculates hashes for potential duplicates
   - Forms duplicate groups
5. **Results** → AnalysisResult with statistics and recommendations
6. **Output** → CLI formats and displays results or exports to file

## Performance Optimizations

### 1. Multi-Stage Hashing
```python
if file_size > QUICK_HASH_SIZE * 3:
    # Large file: quick hash first
    quick_hash = hasher.quick_hash(file)
    if has_duplicate_quick_hash:
        full_hash = hasher.hash_file(file)
else:
    # Small file: direct full hash
    full_hash = hasher.hash_file(file)
```

### 2. Database Indexing
```sql
CREATE INDEX idx_files_hash ON files(hash);
CREATE INDEX idx_files_size ON files(size);
CREATE INDEX idx_files_quick_hash ON files(quick_hash);
```

### 3. Batch Operations
```python
def batch_insert_files(self, files: List[FileInfo]):
    cursor.executemany(sql, data)  # Single transaction
```

### 4. Generator-Based Scanning
```python
def scan_directory(self) -> Generator:
    yield (folder_path, files)  # Memory-efficient streaming
```

## Algorithm Complexity

- **Scanning**: O(n) where n = number of files
- **Size Grouping**: O(n)
- **Hashing**: O(m × s) where m = files with duplicate sizes, s = average file size
- **Duplicate Detection**: O(n log n) for sorting + O(n) for grouping
- **Overall**: O(n log n + m × s)

## Testing Strategy

### Unit Tests
- **test_hasher.py**: Hash calculation, cache behavior
- **test_scanner.py**: Directory traversal, pattern matching
- **test_analyzer.py**: Duplicate detection, statistics

### Test Coverage
```python
class TestFileHasher(unittest.TestCase):
    def test_identical_files_same_hash(self):
        # Verify duplicate detection
        
    def test_cache_invalidation_on_mtime_change(self):
        # Verify cache correctness
```

## Error Handling

### Graceful Degradation
```python
try:
    file_info = self.scan_file(file_path)
except (OSError, IOError) as e:
    logger.error(f"Error scanning {file_path}: {e}")
    self.scan_result.add_error(str(e))
    return None  # Continue with other files
```

### Database Transactions
```python
@contextmanager
def get_connection(self):
    try:
        yield self.connection
    except Exception:
        self.connection.rollback()
        raise
    else:
        self.connection.commit()
```

## Extension Points

### 1. New Hash Algorithms
Add to HashAlgorithm enum and algorithm_map in FileHasher.

### 2. New Storage Backends
Implement Database interface for MongoDB, PostgreSQL, etc.

### 3. GUI Integration
Use core modules directly, bypassing CLI:
```python
from src.core.analyzer import DuplicateAnalyzer
analyzer = DuplicateAnalyzer()
result = analyzer.analyze_folder(folder)
```

### 4. Network File Systems
Extend FileScanner with network path handling.

### 5. Cloud Storage
Add plugins for S3, Google Drive, Dropbox scanning.

## Best Practices Implemented

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Config objects passed to constructors
3. **Type Hints**: Full typing for better IDE support and documentation
4. **Context Managers**: Safe resource management for files and database
5. **Logging**: Structured logging throughout the application
6. **Error Recovery**: Continue processing despite individual file errors
7. **Configurability**: Extensive configuration options
8. **Testing**: Comprehensive unit test coverage
9. **Documentation**: Inline documentation and docstrings
10. **Performance**: Optimized algorithms and database queries