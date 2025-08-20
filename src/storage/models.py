"""Data models for PyDeduper."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class FileInfo:
    """Represents information about a single file."""
    id: Optional[int] = None
    path: Path = None
    name: str = ""
    size: int = 0
    hash: Optional[str] = None
    quick_hash: Optional[str] = None
    parent_folder_id: Optional[int] = None
    last_modified: Optional[datetime] = None
    
    def __post_init__(self):
        """Convert string path to Path object if needed."""
        if self.path and not isinstance(self.path, Path):
            self.path = Path(self.path)
    
    @property
    def full_path(self) -> str:
        """Get the full path as a string."""
        return str(self.path) if self.path else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "path": str(self.path) if self.path else None,
            "name": self.name,
            "size": self.size,
            "hash": self.hash,
            "quick_hash": self.quick_hash,
            "parent_folder_id": self.parent_folder_id,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileInfo":
        """Create FileInfo from dictionary."""
        if data.get("last_modified"):
            data["last_modified"] = datetime.fromisoformat(data["last_modified"])
        return cls(**data)


@dataclass
class FolderInfo:
    """Represents information about a folder."""
    id: Optional[int] = None
    path: Path = None
    name: str = ""
    parent_path: Optional[Path] = None
    num_files: int = 0
    total_size: int = 0
    files: List[FileInfo] = field(default_factory=list)
    subfolders: List["FolderInfo"] = field(default_factory=list)
    
    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        if self.path and not isinstance(self.path, Path):
            self.path = Path(self.path)
        if self.parent_path and not isinstance(self.parent_path, Path):
            self.parent_path = Path(self.parent_path)
    
    @property
    def full_path(self) -> str:
        """Get the full path as a string."""
        return str(self.path) if self.path else ""
    
    @property
    def duplication_percentage(self) -> float:
        """Calculate the percentage of files that are duplicated elsewhere."""
        if not self.files or self.num_files == 0:
            return 0.0
        
        duplicated_count = sum(1 for file in self.files if file.hash and hasattr(file, 'is_duplicate') and file.is_duplicate)
        return (duplicated_count / self.num_files) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "path": str(self.path) if self.path else None,
            "name": self.name,
            "parent_path": str(self.parent_path) if self.parent_path else None,
            "num_files": self.num_files,
            "total_size": self.total_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderInfo":
        """Create FolderInfo from dictionary."""
        return cls(**data)


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate files."""
    hash: str
    size: int
    count: int
    files: List[FileInfo] = field(default_factory=list)
    
    @property
    def total_wasted_space(self) -> int:
        """Calculate total wasted space (size * (count - 1))."""
        return self.size * (self.count - 1) if self.count > 1 else 0
    
    @property
    def locations(self) -> List[str]:
        """Get list of all file locations."""
        return [file.full_path for file in self.files]
    
    def get_recommended_keep(self) -> Optional[FileInfo]:
        """Get the recommended file to keep based on various criteria."""
        if not self.files:
            return None
        
        # Sort by: 1) shortest path, 2) earliest modified, 3) alphabetical
        sorted_files = sorted(
            self.files,
            key=lambda f: (
                len(f.full_path),
                f.last_modified or datetime.max,
                f.full_path
            )
        )
        return sorted_files[0]


@dataclass
class ScanResult:
    """Represents the result of a file system scan."""
    total_files: int = 0
    total_folders: int = 0
    total_size: int = 0
    scan_duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add an error message to the scan result."""
        self.errors.append(error)
    
    @property
    def has_errors(self) -> bool:
        """Check if there were any errors during the scan."""
        return len(self.errors) > 0


@dataclass
class AnalysisResult:
    """Represents the result of duplicate analysis."""
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    total_duplicates: int = 0
    total_wasted_space: int = 0
    folder_duplication_stats: Dict[str, float] = field(default_factory=dict)
    
    def calculate_stats(self):
        """Calculate summary statistics."""
        self.total_duplicates = sum(group.count - 1 for group in self.duplicate_groups)
        self.total_wasted_space = sum(group.total_wasted_space for group in self.duplicate_groups)