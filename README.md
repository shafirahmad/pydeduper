# PyDeduper
Duplicate file finder with advanced folder analysis capabilities

## Features

- **Duplicate Detection**: Find duplicate files based on content (SHA256 hash by default)
- **Folder Analysis**: Calculate duplication percentage for each folder
- **Similar Folder Detection**: Find folders with similar content
- **Multiple Hash Algorithms**: Support for MD5, SHA1, SHA256, SHA512
- **Database Persistence**: SQLite database for efficient analysis of large file systems
- **Flexible Output**: Export results to JSON or CSV
- **Performance Optimized**: Quick hash for initial filtering, full hash only when needed
- **Configurable**: Extensive configuration options via command line or environment variables

## Installation

### From Source
```bash
git clone https://github.com/shafirahmad/pydeduper.git
cd pydeduper
pip install -e .
```

### Using pip (once published)
```bash
pip install pydeduper
```

## Usage

### Basic Usage
```bash
# Scan a single directory
python -m src.cli /path/to/directory

# Scan multiple directories
python -m src.cli /path/to/dir1 /path/to/dir2

# With verbose output
python -m src.cli -v /path/to/directory
```

### Advanced Options
```bash
# Use different hash algorithm
python -m src.cli --hash sha512 /path/to/directory

# Export results to JSON
python -m src.cli --export results.json /path/to/directory

# Find folders with >= 70% similarity
python -m src.cli --find-similar 70 /path/to/directory

# Show only folders with >= 50% duplication
python -m src.cli --folder-threshold 50 /path/to/directory

# Ignore hidden files and follow symlinks
python -m src.cli --ignore-hidden --follow-links /path/to/directory
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-d, --debug` | Enable debug output |
| `--hash {md5,sha1,sha256,sha512}` | Hash algorithm to use (default: sha256) |
| `--db PATH` | Database file path (default: data/pydeduper.db) |
| `--no-db` | Don't use database, analyze in memory only |
| `--all` | Show all duplicate groups (not just top 10) |
| `--export PATH` | Export results to file |
| `--export-format {json,csv}` | Export format (default: json) |
| `--min-size BYTES` | Minimum file size to consider |
| `--folder-threshold PCT` | Minimum duplication % to show folders |
| `--find-similar PCT` | Find folders with similarity >= PCT% |
| `--ignore-hidden` | Ignore hidden files and folders |
| `--follow-links` | Follow symbolic links |

## Project Structure

```
pydeduper/
├── src/
│   ├── core/
│   │   ├── scanner.py      # File system traversal
│   │   ├── hasher.py       # Hash calculation
│   │   └── analyzer.py     # Duplicate detection
│   ├── storage/
│   │   ├── database.py     # SQLite operations
│   │   └── models.py       # Data models
│   ├── cli.py              # Command-line interface
│   └── config.py           # Configuration
├── tests/                  # Unit tests
├── data/                   # Database storage
└── README.md
```

## How It Works

1. **Scanning**: Traverses the file system to collect file information
2. **Size Grouping**: Groups files by size (files with unique sizes can't be duplicates)
3. **Quick Hash**: For large files, calculates a quick hash of the first 1KB
4. **Full Hash**: Only calculates full hash for files with matching quick hashes
5. **Analysis**: Identifies duplicate groups and calculates folder statistics
6. **Reporting**: Displays results with wasted space calculations and recommendations

### The problem
The idea for this came about as I have not found a file deduplicator that can let me know the rate of duplication across folders. Most dedupe software is focused on the file itself, and not the directory structure.

In essence, I wish to be able to go into a folder and find "similar" folders, ie those that contain the exact same files or contain some of the same files as in this folder.

In a Camera images folder for example, I can find that some of the images have been copied (and possibly renamed) to another folder. Now deciding whether to delete the duplicated files in which folder is a question that would be difficult to answer if all you have is a file by file comparison. 

### The solution
To show for each folder, a % of files that are duplicated elsewhere, and where those files are.

### Goals/Milestones
1. Given a set of folders, to walk thru folder structure and find all files, save path, filename, filesize
    - dict? pandas? sqliite?
2. Sort by filesize
3. For each filesize, determine hash of some sort
    - filename only
    - md5 or crc of 1st 1k
    - md5 or crc of whole file
    - contents of each file
    - or some combination, like first 1k, skip 2k, get 1k, skip 4k, get 1k, skip 8k etc
4. Sort hashes within each filesize (to determine dupes)
    - run secondary dedupe algo (like full contents comparison)
5. Walk thru folder structure again, and determine folders with high % of duplication
6. Send output to csv or as a batch file (which can be execued on windows)
7. How to determine which dupes to be deleted
    - date is older / newer
    - filename is longer / shorter
    - filename has least/most % of numbers
    - folder has high dupe% (consider deleting all files in folder)
    - shorter/longer path
8. Add a GUI (like midnight commander) - show folders side by side 
    - probaby 3 pane - left page is selected folder, middle pane is where dupes are found, right page is opened dupe folder
    - Use tkinter? pygame? QT ?
9. Test on different scenarios/large no of files/unforeseen file structures
    - what to do about non-standard characters, diff language etc
    - network files
10. Expand to handle other OS like linux/macos (path strings, links, and other quirks)

### Design considerations
For now, optimization is not the focus. There is no need for speed or memory usage to be a factor (might be an issue later, but we leave this for future time).
Also, it is not inherently for finding similar images, or those in different compressions/resolutions/cropped. I am looking at exact contents duplicates (byte by byte), not near matches or lookalikes - file names shoud not matter. 

### Some inspiration / food for thought
https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/meet.2011.14504801013

https://lifehacker.com/the-best-duplicate-file-finder-for-windows-1696492476

https://softwarerecs.stackexchange.com/questions/32145/windows-tool-to-find-duplicate-folders

https://softwarerecs.stackexchange.com/questions/77961/gratis-windows-tool-to-find-similar-but-not-identical-directories

https://www.scootersoftware.com/features.php

https://www.ultraedit.com/support/tutorials-power-tips/ultracompare/find-duplicates.html

### Licence
For now, the licence is "see only, don't touch" aka "Visible Source". As and when it is decided that more contributors or help is needed, then the licence may be opened up.

### Initial Creation Date: 9 March 2021
### Author: Shafir Ahmad 
### Git: http://www.github.com/shafirahmad/pydeduper
